from typing import Optional

# --- ANTIDEPRESSANT ARMS (For Anchor & Prior Trials) ---

SSRI_INGREDIENTS = {
    'fluoxetine', 
    'sertraline', 
    'citalopram', 
    'escitalopram', 
    'paroxetine', 
    'fluvoxamine'
}
SSRI = "SSRI"

SNRI_INGREDIENTS = {
    'venlafaxine', 
    'desvenlafaxine', 
    'duloxetine', 
    'levomilnacipran',
    'milnacipran' 
}
SNRI = "SNRI"

BUPROPION_INGREDIENTS = {
    'bupropion'
}
BUPROPION = "BUPROPION"

MIRTAZAPINE_INGREDIENTS = {
    'mirtazapine'
}
MIRTAZAPINE = "MIRTAZAPINE"

VORTIOXETINE_INGREDIENTS = {
    'vortioxetine'
}
VORTIOXETINE = "VORTIOXETINE"

# Combined set for finding the anchor
ALL_ARM_INGREDIENTS = (
    SSRI_INGREDIENTS | 
    SNRI_INGREDIENTS | 
    BUPROPION_INGREDIENTS | 
    MIRTAZAPINE_INGREDIENTS | 
    VORTIOXETINE_INGREDIENTS
)

ALL_ARMS = {
    SSRI,
    SNRI,
    BUPROPION,
    MIRTAZAPINE,
    VORTIOXETINE
}

# --- AUGMENTATION INGREDIENTS ---

BUSPIRONE_INGREDIENTS = {
    'buspirone'
}
BUSPIRONE = "BUSPIRONE"

LITHIUM_INGREDIENTS = {
    'lithium'
}
LITHIUM = "LITHIUM"

ANTIPSYCHOTIC_INGREDIENTS = {
    'quetiapine', 
    'aripiprazole', 
    'risperidone', 
    'olanzapine', 
    'ziprasidone', 
    'lurasidone', 
    'brexpiprazole', 
    'cariprazine',
    'clozapine',
    'haloperidol',
    'paliperidone'
}
ANTIPSYCHOTIC = "ANTIPSYCHOTIC"

AUGMENTATION_INGREDIENTS = BUSPIRONE_INGREDIENTS | LITHIUM_INGREDIENTS | ANTIPSYCHOTIC_INGREDIENTS

# --- OTHER CLASSES (For Features) ---

NSAID_INGREDIENTS = {
    'ibuprofen', 
    'naproxen', 
    'diclofenac', 
    'meloxicam', 
    'celecoxib', 
    'ketorolac', 
    'indomethacin', 
    'etodolac',
    'aspirin',
    'nabumetone',
    'piroxicam',
    'sulindac'
}
NSAID = "NSAID"

BENZODIAZEPINE_INGREDIENTS = {
    'alprazolam', 
    'lorazepam', 
    'clonazepam', 
    'diazepam', 
    'temazepam',
    'chlordiazepoxide',
    'clorazepate',
    'oxazepam',
    'midazolam',
    'triazolam'
}
BENZODIAZEPINE = "BENZODIAZEPINE"

HYPNOTICS_INGREDIENTS = {
    'zolpidem', 
    'eszopiclone', 
    'zaleplon', 
    'doxepin' 
}
HYPNOTIC = "HYPNOTIC"

# --- MASTER UNION OF ALL ARMS ---

# Maps specific ingredient string -> Class Label
MASTER_INGREDIENTS_MAP = {}

for ingredient in SSRI_INGREDIENTS: MASTER_INGREDIENTS_MAP[ingredient] = SSRI
for ingredient in SNRI_INGREDIENTS: MASTER_INGREDIENTS_MAP[ingredient] = SNRI
for ingredient in BUPROPION_INGREDIENTS: MASTER_INGREDIENTS_MAP[ingredient] = BUPROPION
for ingredient in MIRTAZAPINE_INGREDIENTS: MASTER_INGREDIENTS_MAP[ingredient] = MIRTAZAPINE
for ingredient in VORTIOXETINE_INGREDIENTS: MASTER_INGREDIENTS_MAP[ingredient] = VORTIOXETINE
for ingredient in BUSPIRONE_INGREDIENTS: MASTER_INGREDIENTS_MAP[ingredient] = BUSPIRONE
for ingredient in LITHIUM_INGREDIENTS: MASTER_INGREDIENTS_MAP[ingredient] = LITHIUM
for ingredient in ANTIPSYCHOTIC_INGREDIENTS: MASTER_INGREDIENTS_MAP[ingredient] = ANTIPSYCHOTIC
for ingredient in NSAID_INGREDIENTS: MASTER_INGREDIENTS_MAP[ingredient] = NSAID
for ingredient in BENZODIAZEPINE_INGREDIENTS: MASTER_INGREDIENTS_MAP[ingredient] = BENZODIAZEPINE
for ingredient in HYPNOTICS_INGREDIENTS: MASTER_INGREDIENTS_MAP[ingredient] = HYPNOTIC


# --- HELPER FUNCTIONS ---

def _in_arm(ingredient: str, arm_set: set[str]) -> bool:
    for arm_ingredient in arm_set:
        if arm_ingredient in ingredient.lower():
            return True
    return False

def get_med_arm(ingredient: str) -> Optional[str]:
    """
    Returns the Antidepressant Class (SSRI, SNRI, etc.) for a given ingredient.
    Returns None if the ingredient is not a target antidepressant.
    """
    if _in_arm(ingredient, SSRI_INGREDIENTS):
        return SSRI
    elif _in_arm(ingredient, SNRI_INGREDIENTS):
        return SNRI
    elif _in_arm(ingredient, BUPROPION_INGREDIENTS):
        return BUPROPION
    elif _in_arm(ingredient, MIRTAZAPINE_INGREDIENTS):
        return MIRTAZAPINE
    elif _in_arm(ingredient, VORTIOXETINE_INGREDIENTS):
        return VORTIOXETINE
    return None