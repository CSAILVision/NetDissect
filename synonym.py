'''Contains a list of synonyms to normalize out of ADE20K data.'''

def synonyms(words):
    '''
    Transforms a list of words to a better list of synonyms.
    Preferred names are listed first, and 
    '''
    # Common case: no common or confusing words
    if not any(w in confusing_names or w in common_names for w in words):
        return words
    # The list needs to be changed.  First, accumulate better names.
    full_list = []
    for word in words:
        full_list.extend(common_names.get(word, []))
    full_list.extend(words)
    # Now accumulate confusing names to omit.
    omit = set()
    for word in full_list:
        if word in confusing_names:
            omit.update(confusing_names[word])
    # Now produce a filtered list
    seen = set()
    see = seen.add
    return [w for w in full_list if not (w in omit or w in seen or see(w))]

# These potential confusing names are drawn from aliases created by
# joining ade20k data with opensurfaces.  Keys are more specific names
# that should not be aliased with more generic names in values.

# Read this list as, { 'flowerpot': ['pot'] } "A flowerpot should not be
# called a pot, because that might confuse it with other pots."

# 29 confusing synonyms to avoid
confusing_names ={
'flowerpot': ['pot'],
'toilet': ['throne', 'can'],
'curtain': ['mantle'],
'fabric': ['material'],
'file cabinet': ['file'],
'chest of drawers': ['chest'],
'fire hydrant': ['plug'],
'car': ['machine'],
'closet': ['press'],
'bicycle': ['wheel'],
'brochure': ['folder'],
'filing cabinet': ['file'],
'paper': ['tissue'], # Opensurfaces groups these materials; call it paper.
'exhaust hood': ['hood'],
'blanket': ['cover'],
'carapace': ['shield'],
'cellular phone': ['cell'],
'handbag': ['bag'],
'land': ['soil'],
'sidewalk': ['pavement'],
'poster': ['card', 'bill'],
'paper towel': ['tissue'],
'computer mouse': ['mouse'],
'steering wheel': ['wheel'],
'lighthouse': ['beacon'],
'basketball hoop': ['hoop'],
'bus': ['passenger vehicle'],
'head': ['caput'],
# Do not use left/right/person to qualify body parts
'arm': ['left arm', 'right arm', 'person arm'],
'foot': ['left foot', 'right foot', 'person foot'],
'hand': ['left hand', 'right hand', 'person hand'],
'leg': ['left leg', 'right leg', 'person leg'],
'shoulder': ['left shoulder', 'right shoulder', 'person shoulder'],
'torso': ['person torso'],
'head': ['person head'],
'hair': ['person hair'],
'nose': ['person nose'],
'ear': ['person ear'],
'neck': ['person neck'],
'eye': ['person eye'],
'eyebrow': ['person eyebrow']
}

# These potential synonyms are drawn from the raw ADE20K data, with
# shorter and more common names listed first.  I have manually uncommented
# word pairs that seem unambiguously the same, where the most common uses
# of the second word would allow the first word to be substituted without
# changing meaning.

common_names = {
# We do not distinguish between left+right parts for our purposes.
'left arm': ['arm'],
'right arm': ['arm'],
'left foot': ['foot'],
'right foot': ['foot'],
'left hand': ['hand'],
'right hand': ['hand'],
'left leg': ['leg'],
'right leg': ['leg'],
'left shoulder': ['shoulder'],
'right shoulder': ['shoulder'],
# And we assume that human parts do not need to say 'person'
'person torso': ['torso'],
'person head': ['head'],
'person arm': ['arm'],
'person hand': ['hand'],
'person hair': ['hair'],
'person nose': ['nose'],
'person leg': ['leg'],
'person mouth': ['mouth'],
'person ear': ['ear'],
'person neck': ['neck'],
'person eye': ['eye'],
'person eyebrow': ['eyebrow'],
'person foot': ['foot'],

# why is this word airport-airport?
'airport-airport-s': ['airport-s'],
# This is the preferred spelling for us.
'aeroplane': ['airplane'],
'airplane': ['airplane', 'aeroplane', 'plane'],
'spectacles': ['eyeglasses'],
'windopane': ['window'],  # ade20k calls windows windowpanes.
'dog': ['dog', 'domestic dog', 'canis familiaris'],
'alga': ['algae'],
'bicycle': ['bicycle', 'bike', 'cycle'],
'food': ['food', 'solid food'],
'caput': ['head'],
'route': ['road'],
'fencing': ['fence'],
'flooring': ['floor'],
'carpet': ['carpet', 'carpeting'],
'shrub': ['bush'],
'armour': ['armor'],
'pail': ['bucket'],
'spigot': ['faucet'],
'faucet': ['faucet', 'spigot'],
'crt screen': ['screen'],
'cistern': ['water tank'],
'video display': ['display'],
'lift': ['elevator'],
'hydroplane': ['seaplane'],
'microwave oven': ['microwave'],
'falls': ['waterfall'],
'mike': ['microphone'],
'windscreen': ['windshield'],
'fluorescent fixture': ['fluorescent'],
'water vapour': ['water vapor'],
'numberplate': ['license plate'],
'tin can': ['can'],
'cow': ['cow', 'moo-cow'],
'horse': ['horse', 'equus caballus'],
'kerb': ['curb'],
'filing cabinet': ['file cabinet'],
'electrical switch': ['switch'],
'telephone set': ['telephone'],
'totaliser': ['adding machine'],
'television receiver': ['television'],
'fabric': ['fabric', 'cloth', 'textile'],
'textile': ['fabric'],
'attack aircraft carrier': ['aircraft carrier'],
'cooking stove': ['stove'],
'electric-light bulb': ['light bulb'],
}
