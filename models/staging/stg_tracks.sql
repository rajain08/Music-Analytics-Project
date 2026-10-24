select
    track_id,
    artist,
    track,
    album,
    case
        when trim(genre) = 'alternative rock.' then 'Alternative Rock'
        when trim(genre) = 'Meta' then 'Metal'
        when trim(genre) = 'deathcore' then 'Deathcore'
        else trim(genre)
    end as primary_genre,
    case
        when try_cast(sentiment_score as double) is not null then mood_category
        else sentiment_score
    end as mood_category,
    case
        when try_cast(sentiment_score as double) is not null then try_cast(sentiment_score as double)
        else try_cast(mood_category as double)
    end as sentiment_score,
    sentiment_polarity,
    try_cast(release_date as date) as release_date
from {{ source('raw', 'raw_tracks') }}
