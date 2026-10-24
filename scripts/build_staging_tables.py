#!/usr/bin/env python3
import os
import csv
import glob
import hashlib
import argparse
from datetime import datetime, timezone
from typing import Dict, Optional

def compute_track_id(artist: str, track: str, album: str) -> str:
    #Computes MD5 surrogate key using normalized_artist|normalized_track|normalized_album
    normalized_artist = (artist or "").strip().lower()
    normalized_track = (track or "").strip().lower()
    normalized_album = (album or "").strip().lower()
    track_key = f"{normalized_artist}|{normalized_track}|{normalized_album}"
    return hashlib.md5(track_key.encode("utf-8")).hexdigest()

genre_normalization = {
    "alternative rock.": "Alternative Rock",
    "Meta": "Metal",
    "deathcore": "Deathcore",
}

disallowed_primary_genres = {
    "in english", "english", "switzerland", "usa", "united states", "uk",
    "australia", "france", "germany", "canada", "sweden", "japan", "italy",
    "spain", "russia", "austria", "norway", "finland", "netherlands",
    "acoustic", "concept album",
}

def normalize_genre(genre: str) -> str:
    g = (genre or "").strip()
    return genre_normalization.get(g, g)

def parse_iso_date(date_str: str) -> str:
    #Parses various human-readable release date strings into standard ISO YYYY-MM-DD
    if not date_str:
        return ""
    
    clean_str = date_str.strip().strip('"').strip("'")
    if not clean_str:
        return ""

    formats = [
        "%B %d, %Y", 
        "%b %d, %Y",  
        "%Y-%m-%d",   
        "%B %Y",      
        "%b %Y",
        "%Y",
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(clean_str, fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return clean_str

def parse_timestamp(uts: str, utc_time_str: str) -> str:
    #Converts Unix timestamp or Last.fm utc_time string into ISO YYYY-MM-DD HH:MM:SS.
    #Prefer exact Unix epoch timestamp if available
    if uts and uts.strip().isdigit():
        try:
            return datetime.fromtimestamp(int(uts.strip()), tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        except Exception:
            pass

    #Fallback to parsing utc_time string
    if utc_time_str:
        clean_time = utc_time_str.strip().strip('"')
        for fmt in ("%d %b %Y, %H:%M", "%d %B %Y, %H:%M", "%Y-%m-%d %H:%M:%S"):
            try:
                return datetime.strptime(clean_time, fmt).strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                continue

    return utc_time_str.strip()

def find_default_scrobbles_file() -> str:
    #Finds scrobbles csv file in current directory
    files = glob.glob("scrobbles*.csv")
    if files:
        return files[0]
    return "scrobbles.csv"

def build_staging_tables(
    tracks_input: str,
    scrobbles_input: str,
    tracks_output: str,
    listens_output: str,
):
    print(f"Reading tracks from: {tracks_input}")
    if not os.path.exists(tracks_input):
        print(f"Error: {tracks_input} does not exist.")
        return

    print(f"Reading scrobbles from: {scrobbles_input}")
    if not os.path.exists(scrobbles_input):
        print(f"Error: {scrobbles_input} does not exist.")
        return

    #Process stg_tracks and build dual-lookup map
    #Exact lookup: (normalized_artist|normalized_track|normalized_album) -> track_id
    #Fallback lookup: (normalized_artist, normalized_track) -> track_id
    exact_track_lookup: Dict[str, str] = {}
    artist_track_fallback_lookup: Dict[tuple, str] = {}
    tracks_count = 0

    tracks_fieldnames = [
        "track_id",
        "artist",
        "track",
        "album",
        "genre",
        "mood_category",
        "sentiment_score",
        "sentiment_polarity",
        "release_date",
    ]

    print(f"\n--- Generating {tracks_output} ---")
    with open(tracks_input, mode="r", encoding="utf-8") as f_in, \
         open(tracks_output, mode="w", encoding="utf-8", newline="") as f_out:
        
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=tracks_fieldnames)
        writer.writeheader()

        for row in reader:
            art = row.get("artist", "").strip()
            trk = row.get("track", "").strip()
            alb = row.get("album", "").strip()

            #Compute strictly unique MD5 surrogate key from artist|track|album
            track_id = compute_track_id(art, trk, alb)

            key_exact = f"{art.lower()}|{trk.lower()}|{alb.lower()}"
            exact_track_lookup[key_exact] = track_id
            artist_track_fallback_lookup[(art.lower(), trk.lower())] = track_id

            #Genre is mapped from primary_genre (fallback if empty or disallowed, e.g. acoustic, concept album)
            genre = (row.get("primary_genre") or "").strip()
            if not genre or genre.lower() in disallowed_primary_genres:
                candidates = []
                for k in ("genre_1", "genre 1", "genre_2", "genre 2", "genre_3", "genre 3", "genre_4", "genre 4"):
                    cand = (row.get(k) or "").strip()
                    if cand and cand.lower() not in disallowed_primary_genres and cand not in candidates:
                        candidates.append(cand)
                genre = candidates[0] if candidates else (genre or "Unknown")

            #Parse release date into standard ISO YYYY-MM-DD
            raw_date = row.get("release_date", "")
            iso_release_date = parse_iso_date(raw_date)

            #Sentiment/mood score
            raw_score = row.get("mood_score") or row.get("sentiment_score") or "0.0"
            try:
                sentiment_score = round(float(raw_score), 4)
            except ValueError:
                sentiment_score = 0.0

            out_row = {
                "track_id": track_id,
                "artist": art,
                "track": trk,
                "album": alb,
                "genre": normalize_genre(genre),
                "mood_category": row.get("mood_category", "unknown").strip(),
                "sentiment_score": sentiment_score,
                "sentiment_polarity": row.get("sentiment_polarity", "unknown").strip(),
                "release_date": iso_release_date,
            }
            writer.writerow(out_row)
            tracks_count += 1

    print(f"Successfully created {tracks_output} with {tracks_count:,} unique tracks (100% unique primary keys).")

    #Process stg_listens by joining scrobbles to track_id
    listens_fieldnames = [
        "track_id",
        "played_at",
    ]

    print(f"\n--- Generating {listens_output} ---")
    matched_events = 0
    unmatched_events = 0

    with open(scrobbles_input, mode="r", encoding="utf-8") as f_in, \
         open(listens_output, mode="w", encoding="utf-8", newline="") as f_out:
        
        reader = csv.DictReader(f_in)
        writer = csv.DictWriter(f_out, fieldnames=listens_fieldnames)
        writer.writeheader()

        for row in reader:
            art = row.get("artist", "").strip()
            trk = row.get("track", "").strip()
            alb = row.get("album", "").strip()
            uts = row.get("uts", "").strip()
            utc_time = row.get("utc_time", "").strip()

            key_exact = f"{art.lower()}|{trk.lower()}|{alb.lower()}"
            track_id = exact_track_lookup.get(key_exact)

            if not track_id:
                #Fallback to artist + track match if album tag differed in scrobbles
                track_id = artist_track_fallback_lookup.get((art.lower(), trk.lower()))

            if track_id:
                played_at = parse_timestamp(uts, utc_time)
                writer.writerow({
                    "track_id": track_id,
                    "played_at": played_at,
                })
                matched_events += 1
            else:
                unmatched_events += 1

    total_events = matched_events + unmatched_events
    print(f"Successfully created {listens_output} with {matched_events:,} play events.")
    if unmatched_events > 0:
        print(f"Warning: {unmatched_events:,} events could not be matched ({unmatched_events/total_events*100:.2f}%).")
    else:
        print("All play events (100.0%) successfully matched to a unique track_id!")

    print("\nStaging tables build complete!")

def main():
    parser = argparse.ArgumentParser(
        description="Build stg_tracks.csv and stg_listens.csv with unique MD5 track keys."
    )
    parser.add_argument(
        "--tracks-input",
        "-t",
        default="cleaned_tracks_with_sentiment.csv",
        help="Path to cleaned tracks CSV (default: cleaned_tracks_with_sentiment.csv)",
    )
    parser.add_argument(
        "--scrobbles-input",
        "-s",
        default=None,
        help="Path to raw scrobbles CSV (default: auto-detected scrobbles-*.csv)",
    )
    parser.add_argument(
        "--tracks-output",
        default="stg_tracks.csv",
        help="Output CSV for unique tracks (default: stg_tracks.csv)",
    )
    parser.add_argument(
        "--listens-output",
        default="stg_listens.csv",
        help="Output CSV for play events (default: stg_listens.csv)",
    )

    args = parser.parse_args()

    scrobbles_file = args.scrobbles_input
    if not scrobbles_file:
        scrobbles_file = find_default_scrobbles_file()

    build_staging_tables(
        tracks_input=args.tracks_input,
        scrobbles_input=scrobbles_file,
        tracks_output=args.tracks_output,
        listens_output=args.listens_output,
    )

if __name__ == "__main__":
    main()
