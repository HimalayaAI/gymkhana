from __future__ import annotations

import re


DEVANAGARI_PLACEHOLDER_PREFIX = "nrt"
EXISTING_ROMAN_PLACEHOLDER_PREFIX = "romterm"
ROMANIZED_OVERRIDE_PLACEHOLDER_PREFIX = "NRTDEV"

DEVANAGARI_TO_ROMANIZED_OVERRIDES = {
    # Technical terms that should stay in the language users expect.
    "एआई": "AI",
    "एपिआई": "API",
    "एपीआई": "API",
    "जीपीयू": "GPU",
    "सीपीयू": "CPU",
    # High-frequency terms where generic romanization is close but unnatural.
    "काठमाडौँमा": "kathmanduma",
    "काठमाडौंमा": "kathmanduma",
    "बैंकहरुले": "bankharule",
    "बैंकहरूले": "bankharule",
    "बैंकहरु": "bankharu",
    "बैंकहरू": "bankharu",
    "बैंकमा": "bankma",
    "नेपालमा": "nepalma",
    "कागजातहरु": "kagajatharu",
    "कागजातहरू": "kagajatharu",
    "कागजात": "kagajat",
    "ग्राहकलाई": "grahaklai",
    "ग्राहक": "grahak",
    "सहायता": "sahayata",
    "अस्पतालमा": "aspatalma",
    "अस्पताल": "aspatal",
    "समाचारमा": "samacharma",
    "समाचार": "samachar",
    "एयरलाइन्स": "airlines",
}


ROMANIZED_TO_DEVANAGARI_OVERRIDES = {
    # Lowercase technical forms should normalize back to protected Latin tokens.
    "cuda": "CUDA",
    "dgx": "DGX",
    "himalayaai": "HimalayaAI",
    "gpu": "GPU",
    "cpu": "CPU",
    "api": "API",
    "ai": "AI",
    "app": "app",
    "website": "website",
    "qa": "QA",
    "rag": "RAG",
    # Reverse high-confidence forms from DEVANAGARI_TO_ROMANIZED_OVERRIDES.
    "nepal": "नेपाल",
    "namaskar": "नमस्कार",
    "bankharule": "बैंकहरुले",
    "bainkaharule": "बैंकहरुले",
    "bankharu": "बैंकहरु",
    "bainkaharu": "बैंकहरु",
    "bankma": "बैंकमा",
    "nepalma": "नेपालमा",
    "kagajatharu": "कागजातहरु",
    "kagajat": "कागजात",
    "grahaklai": "ग्राहकलाई",
    "grahak": "ग्राहक",
    "sahayata": "सहायता",
    "aspatalma": "अस्पतालमा",
    "aspatal": "अस्पताल",
    "samacharma": "समाचारमा",
    "samachar": "समाचार",
    "airlines le": "एयरलाइन्सले",
    "airlines": "एयरलाइन्स",
    "customerle": "customerले",
    "bankharuko": "बैंकहरुको",
    "bankigata": "बैंकिंगगत",
    "banking app": "banking app",
    "mobile banking app": "mobile banking app",
    "digital banking app": "digital banking app",
    "policy document": "policy document",
    "invoice processing": "invoice processing",
    "customer support": "customer support",
    "customer service": "customer service",
    # Common suffix/verb endings that typing engines often miss on round trip.
    "prabandhan": "प्रबन्धन",
    "byabasthapan": "व्यवस्थापन",
    "sanchalan": "सञ्चालन",
    "prashasan": "प्रशासन",
    "nirnaya": "निर्णय",
    "sujhab": "सुझाव",
    "sujhav": "सुझाव",
    "jaankari": "जानकारी",
    "jankari": "जानकारी",
    "pratibedan": "प्रतिवेदन",
    "report": "report",
    "invoice": "invoice",
    "policy": "policy",
    "claim": "claim",
    "claims": "claims",
    "support": "support",
    "help": "help",
    "improve": "improve",
    "training": "training",
    "banking": "banking",
    "bank le": "बैंक ले",
    "bankle": "बैंकले",
    "le": "ले",
    "ra": "र",
    "ma": "मा",
    "ko": "को",
    "ka": "का",
    "ki": "की",
    "lai": "लाई",
    "bata": "बाट",
    "dekhi": "देखि",
    "sanga": "सँग",
    "pani": "पनि",
    "kasari": "कसरी",
    "kina": "किन",
    "kaha": "कहाँ",
    "kahile": "कहिले",
    "prayog": "प्रयोग",
    "garna": "गर्न",
    "gari": "गरी",
    "gare": "गरे",
    "garcha": "गर्छ",
    "garchha": "गर्छ",
    "garchhan": "गर्छन्",
    "garchan": "गर्छन्",
    "hunchha": "हुन्छ",
    "huncha": "हुन्छ",
    "cha": "छ",
    "chha": "छ",
    "chhan": "छन्",
    "sakchha": "सक्छ",
    "sakcha": "सक्छ",
    "sakchhan": "सक्छन्",
    "sakchan": "सक्छन्",
    "garera": "गरेर",
    "badhaune": "बढाउने",
    "paune": "पाउने",
    "jaruri": "जरुरी",
    "chahinchha": "चाहिन्छ",
    "chahincha": "चाहिन्छ",
    "sudhar": "सुधार",
    "sudharna": "सुधार्न",
    "sewa": "सेवा",
    "kendra": "केन्द्र",
    "huna": "हुन",
    "hune": "हुने",
    "hun": "हुन्",
    "chhaina": "छैन",
    "chaina": "छैन",
    "parcha": "पर्छ",
    "parchha": "पर्छ",
    "ramro": "राम्रो",
    "naramro": "नराम्रो",
    "naya": "नयाँ",
    "purano": "पुरानो",
    "khojnuhos": "खोज्नुहोस्",
    "khojnuhosh": "खोज्नुहोस्",
    "kathmanduma": "काठमाडौँमा",
}


PRESERVED_ROMAN_TERMS = (
    "Operational Efficiency",
    "Competitive Advantage",
    "Regulatory Compliance",
    "Infrastructure Development",
    "Customer Experience",
    "Predictive Analytics",
    "Process Automation",
    "Personalized Service",
    "Personalized Financial Product",
    "Investment Suggestion",
    "Customer Satisfaction",
    "Risk Assessment",
    "Risk Management",
    "Credit Scoring",
    "Loan Approval",
    "Financial Data",
    "Social Media",
    "Creditworthiness",
    "Data Privacy",
    "Real-time Monitoring",
    "Suspicious Activity",
    "Customer Behavior",
    "Market Trend",
    "Economic Indicator",
    "Strategic Decision",
    "Document Processing",
    "Data Entry",
    "Compliance Check",
    "Manual Error",
    "Complex Task",
    "Natural Language Processing",
    "Large Language Model",
    "Generative AI",
    "Deep Learning",
    "Neural Network",
    "Digital Banking",
    "Core Banking",
    "Online Banking",
    "Internet Banking",
    "Loan Processing",
    "Insurance Claim",
    "Policy Document",
    "Invoice Processing",
    "Knowledge Base",
    "Support Ticket",
    "Call Center",
    "Contact Center",
    "Mobile Banking App",
    "Mobile Banking",
    "Artificial Intelligence",
    "Machine Learning",
    "Virtual Assistant",
    "Customer Service",
    "Account Statement",
    "Transaction History",
    "Fraud Detection",
    "Historical Data",
    "Banking Sector",
    "banking sector",
    "mobile banking app",
    "mobile banking",
    "operational efficiency",
    "competitive advantage",
    "regulatory compliance",
    "infrastructure development",
    "customer experience",
    "predictive analytics",
    "process automation",
    "personalized service",
    "personalized financial product",
    "investment suggestion",
    "customer satisfaction",
    "risk assessment",
    "risk management",
    "credit scoring",
    "loan approval",
    "financial data",
    "social media",
    "creditworthiness",
    "data privacy",
    "real-time monitoring",
    "suspicious activity",
    "customer behavior",
    "market trend",
    "economic indicator",
    "strategic decision",
    "document processing",
    "data entry",
    "compliance check",
    "manual error",
    "complex task",
    "natural language processing",
    "generative ai",
    "deep learning",
    "neural network",
    "digital banking",
    "core banking",
    "online banking",
    "internet banking",
    "loan processing",
    "insurance claim",
    "policy document",
    "invoice processing",
    "knowledge base",
    "support ticket",
    "call center",
    "contact center",
    "Chatbots",
    "Chatbot",
    "Machine learning",
    "Virtual assistant",
    "Customer service",
    "Account statement",
    "Transaction history",
    "Fraud detection",
    "Historical data",
    "algorithm",
    "Algorithm",
    "probability",
    "Probability",
    "transaction",
    "Transaction",
    "customer",
    "Customer",
    "banking",
    "Banking",
    "security",
    "Security",
    "website",
    "Website",
    "app",
    "App",
    "HimalayaAI",
    "PyTorch",
    "Supabase",
    "CUDA",
    "DGX",
    "GPU",
    "CPU",
    "API",
    "AI",
    "QA",
    "RAG",
)


def contains_devanagari(text: str) -> bool:
    return any("\u0900" <= char <= "\u097f" for char in text)


def english_heavy_terms(text: str) -> tuple[str, ...]:
    terms: list[str] = []
    for term in PRESERVED_ROMAN_TERMS:
        if len(term) < 4:
            continue
        if re.search(rf"(?<![\w]){re.escape(term)}(?![\w])", text, re.IGNORECASE):
            canonical = term.lower()
            if canonical not in terms:
                terms.append(canonical)
    return tuple(terms)


def protect_devanagari_terms(
    text: str,
    overrides: dict[str, str] | None = None,
    *,
    prefix: str = DEVANAGARI_PLACEHOLDER_PREFIX,
) -> tuple[str, dict[str, str]]:
    """Replace known Devanagari terms before phonetic romanization.

    These placeholders are restored after romanization and optional lowercasing,
    so the returned placeholder map must be passed to ``restore_placeholders``.
    """
    replacements = overrides or DEVANAGARI_TO_ROMANIZED_OVERRIDES
    placeholders: dict[str, str] = {}
    protected = text
    for index, (source, replacement) in enumerate(
        sorted(replacements.items(), key=lambda item: -len(item[0]))
    ):
        if source not in protected:
            continue
        placeholder = f" {prefix}{index}term "
        protected = protected.replace(source, placeholder)
        placeholders[placeholder.strip()] = replacement
    return protected, placeholders


def protect_existing_roman_terms(
    text: str,
    terms: tuple[str, ...] = PRESERVED_ROMAN_TERMS,
    *,
    prefix: str = EXISTING_ROMAN_PLACEHOLDER_PREFIX,
) -> tuple[str, dict[str, str]]:
    """Preserve Latin terms that already appear in mixed-script input."""
    placeholders: dict[str, str] = {}
    protected = text
    for index, term in enumerate(sorted(terms, key=len, reverse=True)):
        placeholder = f" {prefix}{index} "
        pattern = re.compile(rf"\b{re.escape(term)}\b")
        if not pattern.search(protected):
            continue
        protected = pattern.sub(placeholder, protected)
        placeholders[placeholder.strip()] = term
    return protected, placeholders


def restore_placeholders(text: str, placeholders: dict[str, str]) -> str:
    """Restore placeholder values after transformations.

    Devanagari romanization may lowercase intermediate text, so restoration checks
    both the original placeholder and its lowercased form.
    """
    restored = text
    for placeholder, value in placeholders.items():
        restored = restored.replace(placeholder.lower(), value)
        restored = restored.replace(placeholder, value)
    return restored


def protect_roman_terms(text: str, terms: tuple[str, ...] = PRESERVED_ROMAN_TERMS) -> str:
    """Wrap preserved Latin terms so romanized-to-Devanagari conversion skips them."""
    protected = text
    for term in sorted(terms, key=len, reverse=True):
        protected = re.sub(
            rf"(?<![{{\w]){re.escape(term)}(?![\w}}])",
            lambda match: "{" + match.group(0) + "}",
            protected,
            flags=re.IGNORECASE,
        )
    return protected


def protect_romanized_overrides(
    text: str,
    overrides: dict[str, str] | None = None,
    *,
    prefix: str = ROMANIZED_OVERRIDE_PLACEHOLDER_PREFIX,
) -> tuple[str, dict[str, str]]:
    """Apply exact romanized overrides before the phonetic fallback runs."""
    replacements = overrides or ROMANIZED_TO_DEVANAGARI_OVERRIDES
    placeholders: dict[str, str] = {}
    protected = text
    for index, (source, replacement) in enumerate(
        sorted(replacements.items(), key=lambda item: -len(item[0]))
    ):
        placeholder = f"{prefix}{index}"
        pattern = re.compile(rf"\b{re.escape(source)}\b", flags=re.IGNORECASE)
        protected = pattern.sub(f"{{{placeholder}}}", protected)
        placeholders[placeholder] = replacement
    return protected, placeholders


def cleanup_spacing(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", text)
    cleaned = re.sub(r"\s+([?.!,।])", r"\1", cleaned)
    cleaned = re.sub(r"([(\[{])\s+", r"\1", cleaned)
    cleaned = re.sub(r"\s+([)\]}])", r"\1", cleaned)
    return cleaned.strip()
