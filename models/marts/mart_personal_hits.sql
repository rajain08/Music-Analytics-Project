{{ config(materialized='table') }}

with dataset_horizon as (
    select max(cast(played_at as date)) as max_catalog_date
    from {{ ref('stg_listens') }}
),

track_listens as (
    select
        track_id,
        count(*) as total_listens,
        min(cast(played_at as date)) as first_played_date,
        max(cast(played_at as date)) as last_played_date
    from {{ ref('stg_listens') }}
    group by track_id
),

tracks as (
    select
        track_id,
        artist,
        track,
        album,
        primary_genre,
        mood_category,
        sentiment_score,
        sentiment_polarity as sentiment,
        release_date
    from {{ ref('stg_tracks') }}
),

track_metrics as (
    select
        t.track_id,
        t.artist,
        t.track,
        t.album,
        t.primary_genre,
        t.mood_category,
        t.sentiment_score,
        t.sentiment,
        t.release_date,
        tl.total_listens,
        tl.first_played_date,
        tl.last_played_date,
        -- Anchor date: use release_date if known and on/before first play, otherwise first_played_date
        case
            when t.release_date is not null and t.release_date <= tl.first_played_date
            then t.release_date
            else tl.first_played_date
        end as start_anchor_date,
        datediff(h.max_catalog_date, case
            when t.release_date is not null and t.release_date <= tl.first_played_date
            then t.release_date
            else tl.first_played_date
        end) as days_since_release,
        datediff(h.max_catalog_date, tl.first_played_date) as days_since_first_listen
    from tracks t
    inner join track_listens tl
        on t.track_id = tl.track_id
    cross join dataset_horizon h
),

track_rates as (
    select
        *,
        greatest(round(days_since_release / 7.0, 2), 1.0) as weeks_since_release,
        round(total_listens / greatest(round(days_since_release / 7.0, 2), 1.0), 3) as plays_per_week_since_release,
        greatest(round(days_since_first_listen / 7.0, 2), 1.0) as weeks_since_first_listen,
        round(total_listens / greatest(round(days_since_first_listen / 7.0, 2), 1.0), 3) as plays_per_week_since_first_listen,
        percent_rank() over (order by total_listens asc) as catalog_play_percentile
    from track_metrics
),

ranked_hits as (
    select
        *,
        case when catalog_play_percentile >= 0.90 then 1 else 0 end as is_personal_hit,
        case
            when catalog_play_percentile >= 0.90 then 'Personal Hit (Top 10%)'
            when catalog_play_percentile >= 0.75 then 'High Performer (Top 25%)'
            when catalog_play_percentile >= 0.40 then 'Mid-Tier'
            else 'Long-Tail'
        end as performance_tier,
        case
            when catalog_play_percentile >= 0.90
            then row_number() over (
                partition by (case when catalog_play_percentile >= 0.90 then 1 else 0 end) 
                order by plays_per_week_since_release desc
            )
            else null
        end as hit_rank_by_weekly_rate
    from track_rates
)

select
    track_id,
    artist,
    track,
    album,
    primary_genre,
    mood_category,
    sentiment,
    sentiment_score,
    release_date,
    first_played_date,
    last_played_date,
    total_listens,
    weeks_since_first_listen,
    plays_per_week_since_first_listen,
    weeks_since_release,
    plays_per_week_since_release,
    round(catalog_play_percentile, 4) as catalog_play_percentile,
    is_personal_hit,
    performance_tier,
    hit_rank_by_weekly_rate
from ranked_hits

