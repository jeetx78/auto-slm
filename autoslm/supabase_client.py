import os
from supabase import create_client

supabase = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_KEY")
)
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)