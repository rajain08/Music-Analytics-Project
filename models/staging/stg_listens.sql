select
    track_id,
    cast(played_at as timestamp) as played_at
from {{ source('raw', 'raw_listens') }}
