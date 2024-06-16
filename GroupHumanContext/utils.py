import torch
from transformers import AutoTokenizer, AutoModel

#get embedding for combined document
def getEmbedding(document,tokenizer, model, max_length=512, stride=256):
    document = "</s>".join(document)
    inputs = tokenizer(document, return_tensors='pt', truncation=True, padding=True)
    input_ids = inputs['input_ids']
    attention_mask = inputs['attention_mask']

    all_outputs = []

    # Slide over the document
    for i in range(0, input_ids.size(1), stride):
        end_index = min(i + max_length, input_ids.size(1))
        chunk_input_ids = input_ids[:, i:end_index]
        chunk_attention_mask = attention_mask[:, i:end_index]

        # Ensure the chunk is of max_length by padding if necessary
        if chunk_input_ids.size(1) < max_length:
            padding_length = max_length - chunk_input_ids.size(1)
            chunk_input_ids = torch.cat([chunk_input_ids, torch.zeros((1, padding_length), dtype=torch.long)], dim=1)
            chunk_attention_mask = torch.cat([chunk_attention_mask, torch.zeros((1, padding_length), dtype=torch.long)], dim=1)

        with torch.no_grad():
            outputs = model(input_ids=chunk_input_ids, attention_mask=chunk_attention_mask)
            all_outputs.append(outputs.last_hidden_state)

    # Concatenate the outputs from all chunks
    concatenated_outputs = torch.stack(all_outputs, dim=0).flatten(start_dim=0, end_dim=-2)
    averaged_outputs = concatenated_outputs.mean(dim=0)

    return averaged_outputs


# Given a ConvoKit conversation, preprocess each utterance's text by tokenizing and truncating using AutoTokenizer
def processDialog(dialog):
    processed = []
    for utterance in dialog.iter_utterances():
        # skip the section header, which does not contain conversational content
        if utterance.meta['is_section_header']:
            continue
        tokens = utterance.text
        # replace out-of-vocabulary tokens
        """for i in range(len(tokens)):
            if tokens[i] not in voc.word2index:
                tokens[i] = "UNK"
        """
        processed.append({"tokens": tokens, "is_attack": int(utterance.meta['comment_has_personal_attack']), "id": utterance.id})
    return processed


#Adapted from ConvoKit tutorial on using CGA corpus
# Load context-reply pairs from the Corpus, filtering to only conversations
# from the specified split (train, val, or test).
# Each conversation, which has N comments (not including the section header) will
# get converted into N-1 comment-reply pairs, one pair for each reply
# (the first comment does not reply to anything).
# Each comment-reply pair is a tuple consisting of the conversational context
# (that is, all comments prior to the reply), the reply itself, the label (that
# is, whether the reply contained a derailment event), and the comment ID of the
# reply (for later use in re-joining with the ConvoKit corpus).
# The function returns a list of such pairs.
def loadPairs(corpus, split=None, last_only=False):
    pairs = []
    for convo in corpus.iter_conversations():
        # consider only conversations in the specified split of the data
        if split is None or convo.meta['split'] == split:
            dialog = processDialog(convo)
            iter_range = range(1, len(dialog)) if not last_only else [len(dialog)-1]
            for idx in iter_range:
                reply = dialog[idx]["tokens"]
                label = dialog[idx]["is_attack"]
                comment_id = dialog[idx]["id"]
                # gather as context all utterances preceding the reply
                context = [u["tokens"] for u in dialog[:idx]]
                pairs.append((context, reply, label, comment_id))
    return pairs