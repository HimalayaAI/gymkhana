"""Human-readable regression cases for Nepali transliteration."""

TRANSLITERATION_CASES = (
    ("devanagari_to_romanized", "नेपाल", "nepal"),
    ("devanagari_to_romanized", "एआई प्रयोग", "AI prayog"),
    ("romanized_to_devanagari", "namaskar", "नमस्कार"),
    ("romanized_to_devanagari", "bank le AI prayog garna sakcha", "बैंक ले AI प्रयोग गर्न सक्छ"),
)

NORMALIZATION_CASES = (
    ("  NEPAL\n", "nepal"),
    ("काठमाण्डौ  नेपाल", "काठमाण्डौ नेपाल"),
    ("API   १२३", "api १२३"),
)
