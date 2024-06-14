from pymer4.models import Lmer

def island_effects_for_model(model_name, control_tuples, island_tuples, construction):
    island_effects = []
    for condition, data in zip(("Simple", "Island"), (control_tuples, island_tuples)):
        for item in data:
            island_effects.append({
                "model": model_name,
                "construction": construction,
                "condition": condition,
                "gap": "+gap",
                "gram": str(item.s_ab),
                "ungram": str(item.s_xb),
                "wh_effect": item.s_ab.critical_surprisal - item.s_xb.critical_surprisal
            })
            island_effects.append({
                "model": model_name,
                "construction": construction,
                "condition": condition,
                "gap": "-gap",
                "gram": str(item.s_xx),
                "ungram": str(item.s_ax),
                "wh_effect": item.s_ax.critical_surprisal - item.s_xx.critical_surprisal
            })
    return island_effects

def modify_base_dict(sentence_key, stim_set, base_dict):
    sent_copy = base_dict.copy()
    # replace 0 with -1 for island
    if sentence_key == "s_ab":
        sent_copy['wh'] = 1
        sent_copy['gap'] = 1
        sent_copy['surprisal'] = stim_set[sentence_key]['critical_surprisal']
    elif sentence_key == "s_xb":
        sent_copy['wh'] = 0
        sent_copy['gap'] = 1
        sent_copy['surprisal'] = stim_set[sentence_key]['critical_surprisal']
    elif sentence_key == "s_ax":
        sent_copy['wh'] = 1
        sent_copy['gap'] = 0
        sent_copy['surprisal'] = stim_set[sentence_key]['critical_surprisal']
    else: # s_xx
        sent_copy['wh'] = 0
        sent_copy['gap'] = 0
        sent_copy['surprisal'] = stim_set[sentence_key]['critical_surprisal']
    return sent_copy

def island_surprisals(condition_name, tuples, model_name, construction_name):
    stim_id = 1
    # adding surprisals individually because that's what works with the sentence tuples
    surprisals = []
    for stim_set in tuples:
        stim_set = stim_set.to_dict()
        base_info = {
            "item": stim_id,
            "model": model_name,
            "island": 1 if condition_name == "island" else 0,
            "construction": construction_name
        }
        surprisals.append(modify_base_dict("s_ab", stim_set, base_info))
        surprisals.append(modify_base_dict("s_xb", stim_set, base_info))
        surprisals.append(modify_base_dict("s_ax", stim_set, base_info))
        surprisals.append(modify_base_dict("s_xx", stim_set, base_info))
        stim_id +=1
    return surprisals

def fit_regression_model(formula, lm_name, condition, surprisal_data):
    condition_data = surprisal_data[(surprisal_data['model'] == lm_name) & (surprisal_data['construction'] == condition)]
    model = Lmer(formula, data = condition_data)
    model.fit()
    return model.summary()

def interaction_effects(formula, conditions, models, surprisal_data):
    # this is fit for island conditions, ideally fit effects of interest in the notebook
    interaction_results = []
    for model in models:
        for condition in conditions:
            summary = fit_regression_model(formula, model, condition, surprisal_data)
            def interactions_at_index(effect_index, effect_label):
                result = summary[['Estimate', 'P-val', 'Sig']].iloc[effect_index]
                result['model'] = model
                result['condition'] = condition
                result['interaction_type'] = effect_label
                return result
            interaction_results.append(interactions_at_index(4, "filler_gap"))
            interaction_results.append(interactions_at_index(-1, "island_filler_gap"))
    return interaction_results
