import surprisal

wh = 'I know what with gusto our uncle grabbed the food in front of the guests at the holiday party .'
that = 'I know that with gusto our uncle grabbed the food in front of the guests at the holiday party .'

print(surprisal.gpt2_surprisal(wh, surprisal.gpt2, surprisal.grnn_hf_tokenizer, surprisal.gpt2_vocab))
print(surprisal.gpt2_surprisal(that, surprisal.gpt2, surprisal.grnn_hf_tokenizer, surprisal.gpt2_vocab))
