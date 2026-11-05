"""
Maps a predicted skin condition to general, non-prescriptive product
*categories* (not specific medical treatments). This is NOT medical advice
-- it's meant to point someone toward the general aisle/category of
over-the-counter product people commonly look for, with a strong nudge
toward seeing a dermatologist for anything serious.

Conditions like SkinCancer, Vasculitis, Lupus, Bullous, and Vascular_Tumors
deliberately have no product suggestions -- these require professional
medical evaluation, and suggesting retail products for them would be
irresponsible.
"""

NO_PRODUCT_DISCLAIMER = (
    "This result can indicate a condition that needs professional medical "
    "evaluation. Please see a dermatologist or doctor rather than "
    "self-treating with over-the-counter products."
)

RECOMMENDATIONS = {
    "Acne": {
        "categories": ["Salicylic acid cleanser", "Benzoyl peroxide spot treatment", "Oil-free moisturizer", "Non-comedogenic sunscreen"],
        "note": "Mild acne often responds to consistent OTC routines. Persistent or cystic acne should be seen by a dermatologist.",
    },
    "Actinic_Keratosis": {
        "categories": [],
        "note": "Actinic keratosis can be precancerous. " + NO_PRODUCT_DISCLAIMER,
    },
    "Benign_tumors": {
        "categories": [],
        "note": NO_PRODUCT_DISCLAIMER,
    },
    "Bullous": {
        "categories": [],
        "note": NO_PRODUCT_DISCLAIMER,
    },
    "Candidiasis": {
        "categories": ["OTC antifungal cream (clotrimazole/miconazole)", "Breathable cotton clothing", "Keep area dry"],
        "note": "If it doesn't improve within 1-2 weeks of OTC antifungal use, see a doctor.",
    },
    "DrugEruption": {
        "categories": [],
        "note": "This may be a reaction to medication. " + NO_PRODUCT_DISCLAIMER,
    },
    "Eczema": {
        "categories": ["Fragrance-free emollient/moisturizer", "Colloidal oatmeal cream", "Mild hydrocortisone cream (short-term)", "Gentle, soap-free cleanser"],
        "note": "Avoid hot showers and known triggers. See a dermatologist if it's widespread or not improving.",
    },
    "Infestations_Bites": {
        "categories": ["Anti-itch cream (hydrocortisone or calamine)", "Antihistamine (oral, OTC)", "Cold compress"],
        "note": "If you suspect scabies/lice or symptoms spread, see a doctor for prescription treatment.",
    },
    "Lichen": {
        "categories": ["Fragrance-free moisturizer", "Mild topical steroid (as advised by a doctor)"],
        "note": "Lichen planus and related conditions usually need a dermatologist's diagnosis and treatment plan.",
    },
    "Lupus": {
        "categories": [],
        "note": NO_PRODUCT_DISCLAIMER,
    },
    "Moles": {
        "categories": [],
        "note": "Most moles are harmless, but any new or changing mole should be checked by a dermatologist.",
    },
    "Psoriasis": {
        "categories": ["Coal tar or salicylic acid shampoo/cream", "Thick fragrance-free moisturizer", "Mild topical steroid (as advised by a doctor)"],
        "note": "Psoriasis is a chronic condition best managed with a dermatologist's treatment plan.",
    },
    "Rosacea": {
        "categories": ["Gentle, fragrance-free cleanser", "Mineral sunscreen (SPF 30+)", "Niacinamide-based moisturizer"],
        "note": "Avoid known triggers (heat, alcohol, spicy food). See a dermatologist for persistent flushing/bumps.",
    },
    "Seborrh_Keratoses": {
        "categories": [],
        "note": "Usually benign, but a dermatologist should confirm the diagnosis if it's new or changing.",
    },
    "SkinCancer": {
        "categories": [],
        "note": "This result may indicate skin cancer. Please see a dermatologist as soon as possible for proper evaluation.",
    },
    "Sun_Sunlight_Damage": {
        "categories": ["Broad-spectrum sunscreen (SPF 30+)", "Vitamin C serum", "Retinol (nighttime use)", "Hydrating moisturizer"],
        "note": "Daily sunscreen is the most effective way to prevent further damage.",
    },
    "Tinea": {
        "categories": ["OTC antifungal cream (terbinafine/clotrimazole)", "Keep area clean and dry", "Avoid sharing towels/clothing"],
        "note": "See a doctor if it doesn't clear up in 2-4 weeks or covers a large area.",
    },
    "Unknown_Normal": {
        "categories": ["General moisturizer", "Daily sunscreen"],
        "note": "No specific concern detected. Maintain a basic skincare routine.",
    },
    "Vascular_Tumors": {
        "categories": [],
        "note": NO_PRODUCT_DISCLAIMER,
    },
    "Vasculitis": {
        "categories": [],
        "note": NO_PRODUCT_DISCLAIMER,
    },
    "Vitiligo": {
        "categories": ["Broad-spectrum sunscreen (depigmented skin burns easily)", "Cosmetic camouflage makeup (optional)"],
        "note": "Treatment options exist (topical/light therapy) -- a dermatologist can advise on what fits your case.",
    },
    "Warts": {
        "categories": ["OTC salicylic acid wart treatment", "Avoid picking/scratching"],
        "note": "If warts persist, spread, or are on the face/genitals, see a doctor rather than self-treating.",
    },
}


def get_recommendation(predicted_class: str) -> dict:
    """Returns a dict with 'categories' (list) and 'note' (str) for a predicted class."""
    return RECOMMENDATIONS.get(
        predicted_class,
        {"categories": [], "note": "No recommendation data available for this class."},
    )
