'''Functionality for a CounterFact dataset that keeps track of which tokens
are subject tokens and which ones are non-subject tokens.
'''

# >>> asdf[0]
# {'case_id': 0, 'pararel_idx': 2796, 'requested_rewrite': {'prompt': 'The mother tongue of {} is', 'relation_id': 'P103', 'target_new': {'str': 'English', 'id': 'Q1860'}, 'target_true': {'str': 'French', 'id': 'Q150'}, 'subject': 'Danielle Darrieux'}, 'paraphrase_prompts': ['Shayna does this and Yossel goes still and dies. Danielle Darrieux, a native', 'An album was recorded for Capitol Nashville but never released. Danielle Darrieux spoke the language'], 'neighborhood_prompts': ['The mother tongue of Léon Blum is', 'The native language of Montesquieu is', 'François Bayrou, a native', 'The native language of Raymond Barre is', 'Michel Rocard is a native speaker of', 'Jacques Chaban-Delmas is a native speaker of', 'The native language of François Bayrou is', 'Maurice Genevoix, speaker of', 'The mother tongue of François Bayrou is', 'Melchior de Vogüé, speaker of'], 'attribute_prompts': ['J.\xa0R.\xa0R. Tolkien is a native speaker of', 'The mother tongue of Douglas Adams is', 'The mother tongue of Paul McCartney is', 'Elvis Presley is a native speaker of', 'Barack Obama, speaker of', 'Douglas Adams, speaker of', 'Meryl Streep, a native', 'George Orwell spoke the language', 'George Washington, a native', 'Michael Jackson, a native'], 'generation_prompts': ["Danielle Darrieux's mother tongue is", 'Where Danielle Darrieux is from, people speak the language of', "Danielle Darrieux's mother tongue is", 'Danielle Darrieux was born in', "Danielle Darrieux's mother tongue is", "Danielle Darrieux's mother tongue is", 'Danielle Darrieux was born in', 'Where Danielle Darrieux is from, people speak the language of', 'Danielle Darrieux was born in', 'Danielle Darrieux was born in']}

class NoShufCounterFact(DocDataset):
    def __init__(self, model, tokenizer, model_name, layer_name, target_idx, dataset_csv, cache_dir, window_size, device):
        super().__init__(model, tokenizer, model_name, layer_name, target_idx, dataset_csv, cache_dir, window_size, device)
        # TODO save the counterfact dataset as a csv with doc, decoded_prefix so that it works here, 
        # as well as with a "entity mask" that saves which tokens are subject/target

    def __getitem__(self, index):
        doc_idx, hidden_states, targets = super().__getitem__(index)
        self.dataset_csv.iloc[doc_idx]
        