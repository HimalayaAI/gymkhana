from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransliterationCase:
    name: str
    source: str
    expected: str
    origin: str


DEVANAGARI_TO_ROMANIZED_CASES = (
    TransliterationCase(
        name="readme_word_matsyendranath",
        source="मत्स्येन्द्रनाथ",
        expected="matsyendranath",
        origin="isDipesh/nepali-romanization README",
    ),
    TransliterationCase(
        name="readme_phrase_maya",
        source="कसैलाई माया गर्ने एउटा भूल गरें मैले",
        expected="kasailai maya garne euta bhul gare maile",
        origin="isDipesh/nepali-romanization README",
    ),
    TransliterationCase(
        name="hardcoded_kathmandu",
        source="काठमाडौँ",
        expected="kathmandu",
        origin="isDipesh/nepali-romanization HARD_CODED",
    ),
    TransliterationCase(
        name="ambiguous_day",
        source="दिन",
        expected="din",
        origin="isDipesh/nepali-romanization HARD_CODED",
    ),
    TransliterationCase(
        name="enterprise_customer_support",
        source="ग्राहक सहायता",
        expected="grahak sahayata",
        origin="project fixture",
    ),
    TransliterationCase(
        name="technical_ai",
        source="एआई प्रयोग",
        expected="AI prayog",
        origin="technical-term protector",
    ),
    TransliterationCase(
        name="banking_prompt",
        source="नेपालमा बैंकहरुले एआई कसरी प्रयोग गर्न सक्छन्?",
        expected="nepalma bankharule AI kasari prayog garna sakchhan?",
        origin="project fixture",
    ),
    TransliterationCase(
        name="hospital_ai",
        source="अस्पतालमा एआई प्रयोग",
        expected="aspatalma AI prayog",
        origin="project fixture",
    ),
    TransliterationCase(
        name="bank_api",
        source="बैंकमा एआई API प्रयोग हुन्छ",
        expected="bankma AI API prayog hunchha",
        origin="technical-term protector",
    ),
    TransliterationCase(
        name="newspaper_documents",
        source="समाचार कागजातहरु",
        expected="samachar kagajatharu",
        origin="project fixture",
    ),
    TransliterationCase(
        name="kathmandu_customer_center",
        source="काठमाडौँमा ग्राहक सहायता केन्द्र छ",
        expected="kathmanduma grahak sahayata kendra chha",
        origin="project fixture",
    ),
    TransliterationCase(
        name="technical_stack_action",
        source="HimalayaAI API ले GPU प्रयोग गर्छ",
        expected="HimalayaAI API le GPU prayog garcha",
        origin="technical-term protector",
    ),
    TransliterationCase(
        name="service_improvement",
        source="ग्राहक सेवा सुधार गर्न सक्छ",
        expected="grahak sewa sudhar garna sakchha",
        origin="project fixture",
    ),
)


ROMANIZED_TO_DEVANAGARI_CASES = (
    TransliterationCase(
        name="nepal",
        source="nepal",
        expected="नेपाल",
        origin="Nepaile-Unicode local smoke case",
    ),
    TransliterationCase(
        name="namaskar",
        source="namaskar",
        expected="नमस्कार",
        origin="Nepaile-Unicode smart converter",
    ),
    TransliterationCase(
        name="technical_ai_sentence",
        source="bank le AI prayog garna sakcha",
        expected="बैंक ले AI प्रयोग गर्न सक्छ",
        origin="technical-term protector",
    ),
    TransliterationCase(
        name="technical_stack_sentence",
        source="HimalayaAI API le GPU prayog garcha",
        expected="HimalayaAI API ले GPU प्रयोग गर्छ",
        origin="technical-term protector",
    ),
    TransliterationCase(
        name="enterprise_customer_support",
        source="grahak sahayata",
        expected="ग्राहक सहायता",
        origin="deterministic override",
    ),
    TransliterationCase(
        name="enterprise_hospital_ai",
        source="aspatalma AI prayog",
        expected="अस्पतालमा AI प्रयोग",
        origin="deterministic override",
    ),
    TransliterationCase(
        name="enterprise_newspaper_documents",
        source="samachar kagajatharu",
        expected="समाचार कागजातहरु",
        origin="deterministic override",
    ),
    TransliterationCase(
        name="customer_need",
        source="grahaklai sahayata chahinchha",
        expected="ग्राहकलाई सहायता चाहिन्छ",
        origin="deterministic override",
    ),
    TransliterationCase(
        name="airlines_suffix",
        source="airlines le grahak sewa sudhar garna sakchha",
        expected="एयरलाइन्सले ग्राहक सेवा सुधार गर्न सक्छ",
        origin="deterministic override",
    ),
    TransliterationCase(
        name="lowercase_api",
        source="bankma ai api prayog hunchha",
        expected="बैंकमा AI API प्रयोग हुन्छ",
        origin="technical-term protector",
    ),
    TransliterationCase(
        name="poolside_mixed_english_terms",
        source="AI (Artificial Intelligence) le customer service, Chatbot, Fraud Detection, Machine learning algorithm, transaction history ra account statement ma help garcha",
        expected="AI (Artificial Intelligence) ले customer service, Chatbot, Fraud Detection, Machine learning algorithm, transaction history र account statement मा help गर्छ",
        origin="mixed English-term preservation fixture",
    ),
    TransliterationCase(
        name="poolside_long_domain_terms",
        source="AI le customer experience improve gari, operational efficiency badhaune, ra competitive advantage paune sakchhan. Data privacy, security, regulatory compliance, training ra infrastructure development jaruri cha.",
        expected="AI ले customer experience improve गरी, operational efficiency बढाउने, र competitive advantage पाउने सक्छन्। Data privacy, security, regulatory compliance, training र infrastructure development जरुरी छ।",
        origin="mixed English-term preservation fixture",
    ),
    TransliterationCase(
        name="kathmandu_suffix",
        source="kathmanduma grahak sahayata kendra chha",
        expected="काठमाडौँमा ग्राहक सहायता केन्द्र छ",
        origin="project fixture",
    ),
    TransliterationCase(
        name="search_documents",
        source="samachar kagajatharu khojnuhos",
        expected="समाचार कागजातहरु खोज्नुहोस्",
        origin="deterministic override",
    ),
    TransliterationCase(
        name="protected_policy_workflow",
        source="policy document ko invoice processing app ma QA chahinchha",
        expected="policy document को invoice processing app मा QA चाहिन्छ",
        origin="protected workflow fixture",
    ),
    TransliterationCase(
        name="digital_banking_support",
        source="digital banking app ma customer support ramro huna parcha",
        expected="digital banking app मा customer support राम्रो हुन पर्छ",
        origin="protected workflow fixture",
    ),
)


ROUND_TRIP_CASES = (
    TransliterationCase(
        name="ai_banking_prompt",
        source="नेपालमा बैंकहरुले एआई कसरी प्रयोग गर्न सक्छन्?",
        expected="नेपालमा बैंकहरुले AI कसरी प्रयोग गर्न सक्छन्?",
        origin="deterministic round trip",
    ),
    TransliterationCase(
        name="technical_ai",
        source="एआई प्रयोग",
        expected="AI प्रयोग",
        origin="Measured adapter round trip",
    ),
    TransliterationCase(
        name="service_improvement",
        source="ग्राहक सेवा सुधार गर्न सक्छ",
        expected="ग्राहक सेवा सुधार गर्न सक्छ",
        origin="deterministic round trip",
    ),
)
