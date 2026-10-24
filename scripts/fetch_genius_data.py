import re
import csv
import time
import requests
from bs4 import BeautifulSoup
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

headers = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

def slugify(text: str) -> str:
    #Formats artist/track to make it URL-friendly
    if not text:
        return ""
    #Replace & with and
    s = text.replace("&", "and")
    #Remove apostrophes, quotes, parentheses, brackets, symbols, punctuation
    s = re.sub(r"[\'\’\`\"\,\.\!\?\:\;\(\)\[\]\{\}\#\$\%\^\*\+\=\/\\\|\<\>]", "", s)
    #Replace any spaces, underscores, or multiple hyphens with a single hyphen
    s = re.sub(r"[\s\-_]+", "-", s).strip("-")
    return s

def build_genius_url(artist: str, track: str) -> str:
    #Builds genius url
    artist_slug = slugify(artist)
    track_slug = slugify(track)
    return f"https://genius.com/{artist_slug}-{track_slug}-lyrics"

def search_genius_fallback(artist: str, track: str, session: requests.Session) -> Optional[str]:
    #Uses genius multi-search api if direct URL 404s
    try:
        query = f"{artist} {track}"
        res = session.get(
            "https://genius.com/api/search/multi",
            params={"q": query},
            headers=headers,
            timeout=(3.0, 5.0),
        )
        if res.status_code == 200:
            data = res.json()
            sections = data.get("response", {}).get("sections", [])
            for section in sections:
                if section.get("type") == "song":
                    hits = section.get("hits", [])
                    if hits:
                        return hits[0].get("result", {}).get("url")
    except Exception:
        pass
    return None

def extract_release_date(soup: BeautifulSoup) -> str:
    #Modern Genius Credit__Label elements with 'Released on'/'Release Date'
    for label in soup.select("div[class*='Credit__Label']"):
        if "release" in label.get_text(strip=True).lower():
            val_el = label.find_next_sibling() or label.parent.select_one("div[class*='Credit__Contributor']")
            if val_el:
                return val_el.get_text(strip=True)

    #Generic fallback for elements containing 'released on'
    for el in soup.find_all(lambda tag: tag.name in ["span", "div", "p"] and tag.string and "released on" in tag.string.lower()):
        nxt = el.find_next_sibling()
        if nxt:
            return nxt.get_text(strip=True)

    return ""

def fetch_song_data(
    artist: str,
    track: str,
    session: requests.Session,
    use_fallback_search: bool = True,
) -> Tuple[List[str], str, str]:
    #Fetches first 4 genres, lyrics, and release date
    direct_url = build_genius_url(artist, track)
    lyrics = ""
    genres: List[str] = []
    release_date = ""

    try:
        #Connect timeout: 3s, Read timeout: 5s
        res = session.get(direct_url, headers=headers, timeout=(3.0, 5.0))

        #If direct URL fails, attempt fallback search
        if res.status_code == 404 and use_fallback_search:
            fallback_url = search_genius_fallback(artist, track, session)
            if fallback_url:
                res = session.get(fallback_url, headers=headers, timeout=(3.0, 5.0))

        if res.status_code == 200:
            soup = BeautifulSoup(res.text, "html.parser")

            #Extract release date from credits
            release_date = extract_release_date(soup)

            #Remove lyric headers like '4 Contributors Love Me Lyrics'
            for header in soup.select("div[class*='LyricsHeader']"):
                header.decompose()

            #Extract lyrics from all lyric container divs
            lyric_containers = soup.select("div[data-lyrics-container='true']")
            parts = []
            for container in lyric_containers:
                for br in container.find_all("br"):
                    br.replace_with("\n")
                parts.append(container.get_text())
            lyrics = "\n".join(parts).strip()

            #Extract tags/genres
            tag_elements = soup.select("a[href*='/tags/']")
            for tag in tag_elements:
                text = tag.get_text(strip=True)
                if text and text not in genres:
                    genres.append(text)

    except Exception:
        #Ignore network errors, return empty so pipeline never stalls
        pass

    return genres[:4], lyrics, release_date

def extract_unique_tracks(input_file: str) -> List[dict]:
    #Reads and deduplicates original scrobbles csv to keep track metadata and count total plays
    print(f"Reading and deduplicating {input_file}...")
    unique_dict: Dict[Tuple[str, str], dict] = {}

    with open(input_file, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            artist = row.get("artist", "").strip()
            track = row.get("track", "").strip()
            if not artist or not track:
                continue

            key = (artist, track)
            if key not in unique_dict:
                unique_dict[key] = {
                    "artist": artist,
                    "track": track,
                    "album": row.get("album", "").strip(),
                    "play_count": 1,
                    "artist_mbid": row.get("artist_mbid", "").strip(),
                    "album_mbid": row.get("album_mbid", "").strip(),
                    "track_mbid": row.get("track_mbid", "").strip(),
                }
            else:
                unique_dict[key]["play_count"] += 1

    unique_list = list(unique_dict.values())
    print(f"Found {len(unique_list):,} unique tracks in listening history")
    return unique_list

def process_unique_tracks(
    input_file: str,
    output_file: str,
    limit: Optional[int] = None,
    workers: int = 7,
    delay: float = 0.2,
):
    #Reduces csv to unique tracks (~11,600 rows)
    #Fetches genres and lyrics with requests.
    #Streams results directly into new output csv
    unique_tracks = extract_unique_tracks(input_file)

    if limit is not None:
        unique_tracks = unique_tracks[:limit]
        print(f"Processing limited to first {len(unique_tracks):,} tracks.")

    fieldnames = [
        "artist",
        "track",
        "album",
        "release_date",
        "play_count",
        "genre 1",
        "genre 2",
        "genre 3",
        "genre 4",
        "lyrics",
        "artist_mbid",
        "album_mbid",
        "track_mbid",
    ]

    print(f"Fetching {len(unique_tracks):,} songs using {workers} worker threads (delay: {delay}s)...")
    print(f"Streaming results directly to: {output_file}")

    def worker_task(track_item):
        art = track_item["artist"]
        trk = track_item["track"]
        session = requests.Session()
        genres, lyrics, release_date = fetch_song_data(art, trk, session)
        if delay > 0:
            time.sleep(delay)
        return track_item, genres, lyrics, release_date

    #Open the new output CSV and write rows as they finish
    with open(output_file, mode="w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(worker_task, item) for item in unique_tracks]
            pbar = tqdm(total=len(unique_tracks), desc="Scraping Genius", unit="song")

            for future in as_completed(futures):
                track_item, genres, lyrics, release_date = future.result()

                row = {
                    "artist": track_item["artist"],
                    "track": track_item["track"],
                    "album": track_item["album"],
                    "release_date": release_date,
                    "play_count": track_item["play_count"],
                    "genre 1": genres[0] if len(genres) > 0 else "",
                    "genre 2": genres[1] if len(genres) > 1 else "",
                    "genre 3": genres[2] if len(genres) > 2 else "",
                    "genre 4": genres[3] if len(genres) > 3 else "",
                    "lyrics": lyrics,
                    "artist_mbid": track_item["artist_mbid"],
                    "album_mbid": track_item["album_mbid"],
                    "track_mbid": track_item["track_mbid"],
                }

                writer.writerow(row)
                f.flush()
                pbar.update(1)

            pbar.close()

    print(f"Done! Successfully wrote {len(unique_tracks):,} tracks to {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Reduce scrobbles to unique tracks and fill out Genius lyrics and genres."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="scrobbles-RJRockeruno-1791841610.csv",
        help="Input scrobbles CSV file (default: scrobbles-RJRockeruno-1791841610.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="unique_tracks_with_lyrics.csv",
        help="Output CSV for unique tracks (default: unique_tracks_with_lyrics.csv)",
    )
    parser.add_argument(
        "--limit",
        "-l",
        type=int,
        default=None,
        help="Limit number of unique tracks to process (e.g. --limit 50)",
    )
    parser.add_argument(
        "--workers",
        "-w",
        type=int,
        default=3,
        help="Number of concurrent worker threads (default: 3)",
    )
    parser.add_argument(
        "--delay",
        "-d",
        type=float,
        default=0.2,
        help="Polite delay between requests per worker in seconds (default: 0.2)",
    )

    args = parser.parse_args()
    process_unique_tracks(
        args.input,
        args.output,
        limit=args.limit,
        workers=args.workers,
        delay=args.delay,
    )
