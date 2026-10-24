{{ config(materialized='table') }}

with personal_hits as (
    select * from {{ ref('mart_personal_hits') }}
)

select
    primary_genre,
    count(*) as total_tracks,
    sum(is_personal_hit) as personal_hit_count,
    round(sum(is_personal_hit) * 1.0 / count(*), 4) as personal_hit_rate,
    sum(total_listens) as genre_total_plays,
    sum(case when is_personal_hit = 1 then total_listens else 0 end) as genre_hit_plays,
    round(
        sum(case when is_personal_hit = 1 then total_listens else 0 end) * 1.0 / nullif(sum(total_listens), 0),
        4
    ) as hit_play_share,
    round(avg(plays_per_week_since_release), 3) as avg_plays_per_week,
    round(
        avg(case when is_personal_hit = 1 then plays_per_week_since_release else null end),
        3
    ) as hit_avg_plays_per_week
from personal_hits
group by primary_genre
