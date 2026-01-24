import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()  # loads .env if present

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError(
        "SUPABASE_URL or SUPABASE_SERVICE_KEY is missing. "
        "Set them as environment variables."
    )

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
