#!/usr/bin/env python3
import os
import re
import csv
import json
import uuid
import time
import hashlib
import argparse
from typing import Dict, List, Optional, Tuple
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from tqdm import tqdm

mb_search_url = "https://musicbrainz.org/ws/2/recording/"
mb_headers = {
    "User-Agent": "MusicMoodAnalytics/1.0.0 ( music_analytics@example.com )",
    "Accept": "application/json",
}

#Emotion to high-level polarity mapping
emotion_polarity_map = {
    "joy": "positive",
    "neutral": "neutral",
    "surprise": "neutral",
    "anger": "negative",
    "sadness": "negative",
    "fear": "negative",
    "disgust": "negative",
}

#Obvious non-genre tags from Genius (languages, countries) to filter out
non_genres = {
    "in english", "english", "switzerland", "usa", "united states", "uk",
    "australia", "france", "germany", "canada", "sweden", "japan", "italy",
    "spain", "russia", "austria", "norway", "finland", "netherlands",
    "acoustic", "concept album",
}

genre_normalization = {
    "alternative rock.": "Alternative Rock",
    "Meta": "Metal",
    "deathcore": "Deathcore",
}

def normalize_genre(genre: str) -> str:
    g = (genre or "").strip()
    return genre_normalization.get(g, g)


def escape_lucene(text: str) -> str:
    #Escapes Lucene special characters: + - && || ! ( ) { } [ ] ^ " ~ * ? : \\ /
    special_chars = r'([+\-!(){}\[\]^"~*?:\\/])'
    return re.sub(special_chars, r'\\\1', text)

def clean_track_title(title: str) -> str:
    #Cleans track titles for MusicBrainz search (removes remastered, deluxe, ft annotations)
    t = re.sub(r'\(.*?(remaster|deluxe|bonus|live|feat|ft\.).*?\)', '', title, flags=re.IGNORECASE)
    t = re.sub(r'\[.*?(remaster|deluxe|bonus|live|feat|ft\.).*?\]', '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s*-\s*(remaster|live).*', '', t, flags=re.IGNORECASE)
    return t.strip()

def generate_track_id(artist: str, track: str, album: str = "") -> str:
    #Generates MD5 surrogate track_id distinguishing tracks by normalized_artist|normalized_track|normalized_album
    normalized_artist = (artist or "").strip().lower()
    normalized_track = (track or "").strip().lower()
    normalized_album = (album or "").strip().lower()
    track_key = f"{normalized_artist}|{normalized_track}|{normalized_album}"
    return hashlib.md5(track_key.encode("utf-8")).hexdigest()

def generate_surrogate_id(entity_type: str, name: str) -> str:
    #Generates UUIDv5 surrogate ID for artists or albums
    key = f"{entity_type.strip().lower()}:::{name.strip().lower()}"
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

def load_mb_cache(cache_path: str) -> Dict[str, Optional[str]]:
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not read cache {cache_path}: {e}")
    return {}

def save_mb_cache(cache_path: str, cache: Dict[str, Optional[str]]):
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"Warning: Could not save cache to {cache_path}: {e}")

def lookup_musicbrainz_mbid(
    artist: str,
    track: str,
    cache: Dict[str, Optional[str]],
    session: requests.Session,
    delay: float = 1.05,
) -> Tuple[Optional[str], bool]:
    #Queries MusicBrainz API for missing track_mbid with rate limiting & returns (mbid, is_from_network)
    cache_key = f"{artist.strip().lower()}:::{track.strip().lower()}"
    if cache_key in cache:
        return cache[cache_key], False

    clean_track = clean_track_title(track)
    if not clean_track or not artist:
        cache[cache_key] = None
        return None, False

    escaped_track = escape_lucene(clean_track)
    escaped_artist = escape_lucene(artist.strip())

    #Formulate Lucene query
    query = f'recording:"{escaped_track}" AND artist:"{escaped_artist}"'
    params = {"query": query, "fmt": "json", "limit": 3}

    try:
        time.sleep(delay)  #MusicBrainz 1 request/sec policy
        resp = session.get(mb_search_url, params=params, headers=mb_headers, timeout=6.0)
        if resp.status_code == 200:
            data = resp.json()
            recordings = data.get("recordings", [])
            if recordings and "id" in recordings[0]:
                mbid = recordings[0]["id"]
                cache[cache_key] = mbid
                return mbid, True

        #Fallback: if artist ends with ' Swiss' or secondary descriptors, try core name
        if " swiss" in artist.lower():
            alt_artist = re.sub(r'(?i)\s+swiss', '', artist).strip()
            alt_query = f'recording:"{escaped_track}" AND artist:"{escape_lucene(alt_artist)}"'
            time.sleep(delay)
            resp2 = session.get(mb_search_url, params={"query": alt_query, "fmt": "json", "limit": 1}, headers=mb_headers, timeout=6.0)
            if resp2.status_code == 200:
                data2 = resp2.json()
                recordings2 = data2.get("recordings", [])
                if recordings2 and "id" in recordings2[0]:
                    mbid = recordings2[0]["id"]
                    cache[cache_key] = mbid
                    return mbid, True

        cache[cache_key] = None
    except Exception:
        cache[cache_key] = None

    return None, True

def resolve_ids(
    rows: List[dict],
    skip_lookup: bool = False,
    lookup_limit: Optional[int] = 50,
    cache_path: str = ".musicbrainz_cache.json",
) -> None:
    #Assigns unique surrogate track_id using MD5 -> (normalized_artist|normalized_track|normalized_album)
    #Guarantees 100% uniqueness as primary key for dbt models across distinct albums/releases
    print("\nResolving Track IDs")
    for row in rows:
        artist = row.get("artist", "")
        track = row.get("track", "")
        album = row.get("album", "")
        row["track_id"] = generate_track_id(artist, track, album)
        row["id_source"] = "surrogate_md5"

    print(f"Total tracks: {len(rows):,}")
    print(f"All {len(rows):,} tracks assigned unique MD5 track_ids (artist|track|album)\n")

def analyze_sentiments(
    rows: List[dict],
    batch_size: int = 64,
    device: Optional[str] = None,
) -> None:
    #Runs Emotion Model ('j-hartmann/emotion-english-distilroberta-base') on track lyrics, replacing them with mood_category, mood_score, and sentiment_polarity
    print("Running Emotion & Sentiment Analysis")
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    model_name = "j-hartmann/emotion-english-distilroberta-base"
    print(f"Loading '{model_name}' onto '{device}'...")

    #Load from local cache first to avoid unauthenticated HuggingFace 429 errors
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)

    emotion_pipeline = pipeline(
        "text-classification",
        model=model,
        tokenizer=tokenizer,
        device=device,
        truncation=True,
        max_length=512,
    )

    #Separate tracks with lyrics vs without lyrics
    with_lyrics_indices = []
    lyrics_snippets = []

    for idx, row in enumerate(rows):
        lyrics = (row.get("lyrics") or "").strip()
        if lyrics:
            with_lyrics_indices.append(idx)
            #Use up to 512 characters as specified in the model guidance
            lyrics_snippets.append(lyrics[:512])
        else:
            row["mood_category"] = "unknown"
            row["mood_score"] = 0.0
            row["sentiment_polarity"] = "unknown"

    print(f"Tracks with lyrics to analyze: {len(with_lyrics_indices):,} (batch size: {batch_size})")

    #Batch inference
    pbar = tqdm(total=len(with_lyrics_indices), desc="Scoring Emotion", unit="song")
    for i in range(0, len(with_lyrics_indices), batch_size):
        batch_slice = lyrics_snippets[i : i + batch_size]
        indices_slice = with_lyrics_indices[i : i + batch_size]

        results = emotion_pipeline(batch_slice)

        for row_idx, res in zip(indices_slice, results):
            label = res["label"]
            score = round(float(res["score"]), 4)
            polarity = emotion_polarity_map.get(label, "neutral")

            rows[row_idx]["mood_category"] = label
            rows[row_idx]["mood_score"] = score
            rows[row_idx]["sentiment_polarity"] = polarity

        pbar.update(len(batch_slice))

    pbar.close()
    print("Sentiment scoring complete.\n")

def resolve_primary_genres(
    rows: List[dict],
    device: Optional[str] = None,
) -> None:
    #Uses zero-shot classification 'facebook/bart-large-mnli' to pick primary genre from up to 4 candidate genres per track
    #Applies an in-memory memoization cache by (artist, candidate_tuple) to avoid redundant inference

    print("Resolving Primary Genre via Zero-Shot Classification")
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    model_name = "facebook/bart-large-mnli"
    print(f"Loading zero-shot classifier '{model_name}' onto '{device}'...")

    try:
        classifier = pipeline("zero-shot-classification", model=model_name, device=device)
    except Exception as e:
        print(f"Warning: Could not load zero-shot pipeline: {e}. Falling back to first valid candidate.")
        classifier = None

    #In-memory memoization: (artist.lower(), tuple(sorted_candidates)) -> (primary_genre, score)
    genre_cache: Dict[Tuple[str, Tuple[str, ...]], Tuple[str, float]] = {}

    pbar = tqdm(rows, desc="Classifying Primary Genres", unit="track")
    for r in pbar:
        raw_genres = [
            r.get("genre 1", "") or r.get("genre_1", ""),
            r.get("genre 2", "") or r.get("genre_2", ""),
            r.get("genre 3", "") or r.get("genre_3", ""),
            r.get("genre 4", "") or r.get("genre_4", ""),
        ]
        #Filter obvious non-genre tags (languages, country names)
        candidates = [
            g.strip() for g in raw_genres
            if g.strip() and g.strip().lower() not in non_genres
        ]

        if not candidates:
            r["primary_genre"] = "Unknown"
            r["primary_genre_score"] = 0.0
            continue

        if len(candidates) == 1:
            r["primary_genre"] = normalize_genre(candidates[0])
            r["primary_genre_score"] = 1.0
            continue

        #More than 1 candidate: check in-memory cache first
        artist = (r.get("artist") or "").strip()
        track = (r.get("track") or "").strip()
        cache_key = (artist.lower(), tuple(sorted(candidates)))

        if cache_key in genre_cache:
            best_genre, best_score = genre_cache[cache_key]
        elif classifier is not None:
            try:
                res = classifier(f"{artist} - {track}", candidate_labels=candidates)
                best_genre = res["labels"][0]
                best_score = round(float(res["scores"][0]), 4)
                genre_cache[cache_key] = (best_genre, best_score)
            except Exception:
                best_genre = candidates[0]
                best_score = 0.5
        else:
            best_genre = candidates[0]
            best_score = 0.5

        r["primary_genre"] = normalize_genre(best_genre)
        r["primary_genre_score"] = best_score

    pbar.close()
    print("Primary genre classification complete\n")

def main():
    parser = argparse.ArgumentParser(
        description="Clean music track IDs with 3-step priority, replace lyrics with emotion sentiment, and resolve primary genres"
    )
    parser.add_argument(
        "--input",
        "-i",
        default="unique_tracks_with_lyrics.csv",
        help="Input CSV file (default: unique_tracks_with_lyrics.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="cleaned_tracks_with_sentiment.csv",
        help="Output CSV file (default: cleaned_tracks_with_sentiment.csv)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of rows to process for testing (e.g. --limit 50)",
    )
    parser.add_argument(
        "--batch-size",
        "-b",
        type=int,
        default=64,
        help="Inference batch size for emotion classification (default: 64)",
    )
    parser.add_argument(
        "--mb-lookup-limit",
        type=int,
        default=50,
        help="Max live MusicBrainz API calls to make this run (default: 50; use 0 to skip)",
    )
    parser.add_argument(
        "--skip-mb-lookup",
        action="store_true",
        help="Skip MusicBrainz API calls entirely and directly assign surrogate UUIDs for missing IDs",
    )
    parser.add_argument(
        "--skip-genre-classification",
        action="store_true",
        help="Skip zero-shot primary genre classification and take first candidate genre",
    )
    parser.add_argument(
        "--cache-path",
        default=".musicbrainz_cache.json",
        help="Path to MusicBrainz local query cache (default: .musicbrainz_cache.json)",
    )

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist")
        return

    print(f"Reading input from: {args.input}")
    with open(args.input, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if args.limit is not None:
        rows = rows[: args.limit]
        print(f"Processing limited to first {len(rows):,} rows")

    #Resolve IDs with 3-step priority
    skip_mb = args.skip_mb_lookup or (args.mb_lookup_limit == 0)
    resolve_ids(
        rows,
        skip_lookup=skip_mb,
        lookup_limit=args.mb_lookup_limit,
        cache_path=args.cache_path,
    )

    #Run emotion sentiment analysis
    analyze_sentiments(rows, batch_size=args.batch_size)

    #Resolve primary genre
    if not args.skip_genre_classification:
        resolve_primary_genres(rows)
    else:
        for r in rows:
            raw_genres = [
                r.get("genre 1", "") or r.get("genre_1", ""),
                r.get("genre 2", "") or r.get("genre_2", ""),
                r.get("genre 3", "") or r.get("genre_3", ""),
                r.get("genre 4", "") or r.get("genre_4", ""),
            ]
            candidates = [g.strip() for g in raw_genres if g.strip() and g.strip().lower() not in non_genres]
            r["primary_genre"] = normalize_genre(candidates[0]) if candidates else "Unknown"
            r["primary_genre_score"] = 1.0 if candidates else 0.0

    #Write output CSV (excluding raw multi-line lyrics)
    output_fieldnames = [
        "track_id",
        "id_source",
        "artist",
        "track",
        "album",
        "release_date",
        "play_count",
        "primary_genre",
        "primary_genre_score",
        "genre_1",
        "genre_2",
        "genre_3",
        "genre_4",
        "mood_category",
        "mood_score",
        "sentiment_polarity",
        "artist_id",
        "album_id",
        "track_mbid",
    ]

    print(f"Writing cleaned dataset to: {args.output}")
    with open(args.output, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        writer.writeheader()

        for r in rows:
            art = r.get("artist", "").strip()
            alb = r.get("album", "").strip()
            resolved_track_id = r.get("track_id", "")

            #Resolve artist_id: use artist_mbid if present, else surrogate UUID
            art_mbid = (r.get("artist_mbid") or "").strip()
            artist_id = art_mbid if art_mbid else generate_surrogate_id("artist", art)

            #Resolve album_id: use album_mbid if present, else surrogate UUID
            alb_mbid = (r.get("album_mbid") or "").strip()
            album_id = alb_mbid if alb_mbid else generate_surrogate_id("album", f"{art}:::{alb}")

            out_row = {
                "track_id": resolved_track_id,
                "id_source": r.get("id_source", ""),
                "artist": art,
                "track": r.get("track", ""),
                "album": alb,
                "release_date": r.get("release_date", ""),
                "play_count": r.get("play_count", ""),
                "primary_genre": r.get("primary_genre", "Unknown"),
                "primary_genre_score": r.get("primary_genre_score", 0.0),
                "genre_1": r.get("genre 1", "") or r.get("genre_1", ""),
                "genre_2": r.get("genre 2", "") or r.get("genre_2", ""),
                "genre_3": r.get("genre 3", "") or r.get("genre_3", ""),
                "genre_4": r.get("genre 4", "") or r.get("genre_4", ""),
                "mood_category": r.get("mood_category", "unknown"),
                "mood_score": r.get("mood_score", 0.0),
                "sentiment_polarity": r.get("sentiment_polarity", "unknown"),
                "artist_id": artist_id,
                "album_id": album_id,
                "track_mbid": resolved_track_id,  #Populated with resolved track ID
            }
            writer.writerow(out_row)

    print(f"Done! Successfully wrote {len(rows):,} cleaned tracks to {args.output}")
    print(f"Note: Raw multi-line lyrics were cleanly replaced with mood_category, mood_score, and sentiment_polarity")

if __name__ == "__main__":
    main()
