"""
General over-the-counter (OTC) medicine info per condition.

This is educational information only, NOT a prescription. Dosage is
intentionally not included -- always follow the product label or ask a
pharmacist/doctor. Conditions that need medical evaluation have no
medicines listed and tell the user to consult a doctor.
"""

CONFIDENCE_THRESHOLD = 0.50  # below this, treat the result as "unknown"

CONSULT_DOCTOR_MSG = (
    "The model is not confident about this image, or this condition needs a "
    "professional diagnosis. Please consult a doctor or dermatologist. "
    "Do not self-medicate."
)

MEDICINES = {
    "Acne": ["Benzoyl peroxide 2.5-5% gel", "Salicylic acid 2% cleanser", "Adapalene 0.1% gel (OTC in many countries)"],
    "Candidiasis": ["Clotrimazole 1% cream", "Miconazole 2% cream"],
    "Eczema": ["Fragrance-free emollient (ceramide moisturizer)", "Hydrocortisone 1% cream (short-term)", "Cetirizine tablets for itching"],
    "Infestations_Bites": ["Calamine lotion", "Hydrocortisone 1% cream", "Cetirizine or loratadine tablets"],
    "Lichen": ["Fragrance-free moisturizer (for comfort only; a doctor must confirm and treat)"],
    "Psoriasis": ["Coal tar or salicylic acid shampoo/cream", "Thick emollient (urea or petroleum-based)"],
    "Rosacea": ["Azelaic acid 10% cream", "Mineral sunscreen SPF 30+", "Gentle fragrance-free cleanser"],
    "Sun_Sunlight_Damage": ["Broad-spectrum sunscreen SPF 30+", "Aloe vera gel (for burns)", "Moisturizer with vitamin C or niacinamide"],
    "Tinea": ["Terbinafine 1% cream", "Clotrimazole 1% cream", "Antifungal powder for sweaty areas"],
    "Vitiligo": ["Broad-spectrum sunscreen SPF 30+ (a dermatologist can discuss light/topical therapy)"],
    "Warts": ["Salicylic acid 17% wart paint/patches"],
    "Unknown_Normal": ["Daily moisturizer", "Daily sunscreen SPF 30+"],
}

# Always consult a doctor for these, even if the model is confident.
DOCTOR_ONLY = {
    "Actinic_Keratosis", "Benign_tumors", "Bullous", "DrugEruption", "Lupus",
    "Moles", "Seborrh_Keratoses", "SkinCancer", "Vascular_Tumors", "Vasculitis",
}


# Treatments a DOCTOR may prescribe for serious conditions. Shown for
# information only, never as self-treatment. Add or edit lines freely.
DOCTOR_PRESCRIBED = {
    "Actinic_Keratosis": ["Fluorouracil 5% cream (prescription)", "Imiquimod cream (prescription)", "Cryotherapy (freezing) done by a doctor"],
    "Benign_tumors": ["Usually no medicine needed; a doctor may suggest surgical removal or laser if it bothers you"],
    "Bullous": ["Topical clobetasol cream (prescription)", "Oral prednisolone (prescription)"],
    "DrugEruption": ["Antihistamines (e.g. cetirizine) for itching", "Corticosteroids if severe (prescription)", "Stopping the suspected drug only on a doctor's advice"],
    "Lupus": ["Hydroxychloroquine (prescription)", "Topical corticosteroids (prescription)", "Broad-spectrum sunscreen SPF 50+"],
    "Moles": ["No medicine needed; a dermatologist checks it and may remove a suspicious mole", "Broad-spectrum sunscreen SPF 30+"],
    "Seborrh_Keratoses": ["No medicine needed; a doctor can remove it by freezing or curettage if it bothers you"],
    "SkinCancer": ["Treatment is decided by a specialist: surgery, Mohs surgery, or prescribed creams for superficial cases"],
    "Vascular_Tumors": ["Propranolol (prescription, for infantile hemangioma)", "Timolol gel (prescription)"],
    "Vasculitis": ["Oral corticosteroids such as prednisolone (prescription)", "Immunosuppressants such as azathioprine or methotrexate (prescription)"],
}


def get_advice(predicted_class: str, confidence: float) -> dict:
    """
    Returns {'medicines': [...], 'prescribed': [...], 'see_doctor': bool, 'message': str}.
    Low confidence or serious/unknown conditions -> consult a doctor.
    'prescribed' = treatments a doctor may prescribe (information only).
    """
    if confidence < CONFIDENCE_THRESHOLD:
        return {
            "medicines": [],
            "prescribed": [],
            "see_doctor": True,
            "message": "Unknown / low-confidence result. " + CONSULT_DOCTOR_MSG,
        }
    if predicted_class in DOCTOR_ONLY or predicted_class not in MEDICINES:
        return {
            "medicines": [],
            "prescribed": DOCTOR_PRESCRIBED.get(predicted_class, []),
            "see_doctor": True,
            "message": CONSULT_DOCTOR_MSG,
        }

    return {
        "medicines": MEDICINES[predicted_class],
        "prescribed": [],
        "see_doctor": False,
        "message": (
            "These are common OTC options, not a prescription. Follow the label, "
            "check for allergies, and see a doctor if there is no improvement in "
            "1-2 weeks or it gets worse."
        ),
    }