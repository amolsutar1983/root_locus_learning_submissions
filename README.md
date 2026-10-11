# Root Locus Learning Laboratory

A guided, online, mobile-friendly virtual experiment for teaching the Root Locus Method.

## Learning design

Predict → Calculate → Reveal → Simulate → Verify → Reflect.

Students work through:
1. Open-loop poles and zeros
2. Real-axis rule
3. Asymptotes and centroid
4. Breakaway / break-in points
5. Stability range of K
6. Interactive simulation
7. Concept check
8. Digital workbook submission

## Deployment

1. Create/open the same Supabase project used by the other virtual labs.
2. Run `supabase_schema.sql` in Supabase SQL Editor.
3. Upload the project to GitHub.
4. Deploy `app.py` on Streamlit Community Cloud.
5. Add Streamlit Secrets:

```toml
SUPABASE_URL = "https://YOUR_PROJECT.supabase.co"
SUPABASE_KEY = "YOUR_API_KEY"
FACULTY_PASSWORD = "YOUR_FACULTY_PASSWORD"
```

Do not put secrets in GitHub.

## Database

This experiment uses:
`root_locus_learning_submissions`

It does not overwrite the earlier root-locus table.

## Educational note

The numerical breakaway and stability calculations are intended for interactive learning and verification. For an institutional release, add authenticated users, stricter Supabase RLS, auto-grading rubrics, and LMS integration.
