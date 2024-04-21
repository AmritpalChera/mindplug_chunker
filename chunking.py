import numpy as np
import spacy

upper_limit=1024
lower_limit=20




# Load the Spacy model
nlp = spacy.load('en_core_web_lg')
nlp.remove_pipe("lemmatizer")
nlp.add_pipe("lemmatizer", config={"mode": "lookup"}).initialize()

def process(text):
    doc = nlp(text)
    sents = list(doc.sents)
    vecs = np.stack([sent.vector / sent.vector_norm for sent in sents])

    return sents, vecs

def cluster_text(sents, vecs, threshold):
    clusters = [[0]]
    print("\n\n")
    for i in range(1, len(sents)):
        dotted = np.dot(vecs[i], vecs[i-1])
        print('[', dotted, ']: ', sents[i], end=" ")
        if dotted < threshold:
            clusters.append([])
        clusters[-1].append(i)
    print("\n\n")
    return clusters

def clean_text(text):
    # Add your text cleaning process here
    return text

def getElementArrLen (arr):
    print('arrr is: ', arr)
    arrLen = 0
    for i in range (0, len(arr)):
        sectionLength = len(arr[i].split())
        arrLen += sectionLength
    print ('returning arr length is: ', arrLen)
    return arrLen

def combineSentences (sub_divided_sents):
    # merge all indicies with a single element with the next
    to_delete = []
    totalSentences = len(sub_divided_sents)

    # different cases: for each short sentence, could be a list item of previous or starting of a new section.
        # take last sentence of previous chunk
        # first chunk of next chunk
        # if high match, combine to previous chunk, else combine to next chunk
        # repeat this process until no changes to chunks

    for i in range(1, totalSentences):
        prev_sent = sub_divided_sents[i-1] # previous chunk
        curr_division = sub_divided_sents[i] # curr chunk


        prev_division_len = len(prev_sent) # length of previous

        next_sent = None
        if i < totalSentences - 1: next_sent = sub_divided_sents[i+1]

        print("previous division len: ", prev_division_len, ": ", prev_sent)
        # simple case that the previous section only has on chunk. 
        if (prev_division_len == 1):
            sub_divided_sents[i] = prev_sent + curr_division
            to_delete.append(i-1)
        # if there are 2 sentences in the current division and not many words in total
        elif (prev_division_len == 2):
            totalLen = getElementArrLen(prev_sent)
            if (totalLen < 30):
                sub_divided_sents[i] = prev_sent + curr_division
                to_delete.append(i-1)
            # print("possible mergable section", prev_sent)

    # now check if the last index is too short. Combine it with previous
    if len(sub_divided_sents[-1]) == 1 and totalSentences > 1:
        sub_divided_sents[-2] = sub_divided_sents[-2] + sub_divided_sents[-1]
        to_delete.append(-1)

    # delete all merged indicies
    for i in range (len(to_delete)-1, -1 , -1): 
        del sub_divided_sents[to_delete[i]]

    #combine each array into tasks
    combined = []
    for i in range(0, len(sub_divided_sents)):
        combined.append(''.join(sub_divided_sents[i]))

    # for combination in combined:
    #     print('combined: ', combination, "\n")

    return combined

# run this one time during initial run
def run_manual_chunk(cluster_txt):
    sents_div, vecs_div = process(cluster_txt)
   
    print("\n\n")
    # 0th sentence is always the starting index
    section = []
    all_sentences = []

    for i in range(len(sents_div)):
        sent = sents_div[i].text.strip()
        all_sentences.append(sent)
        sentSplit = sent.split()
        if len(sentSplit) < 10:
            # check if the the split sentence represnts a section
            print('short sentence: ', sent)
            section.append(i)

    # combine the sections and manually cluster
    sub_divided_sents = []
    total_sections = len(section)
    print("total short sentences: ", total_sections, ' with indicies: ', section)
    
    for i in range(0, total_sections):
        cur_index = section[i]
        next_index = cur_index
        if i < total_sections -1:
            next_index = section[i+1]
        else: next_index = None
        if (next_index):
            sub_divided_sents += [all_sentences[cur_index: next_index]]
        else:
            sub_divided_sents += [all_sentences[cur_index:]]

    if section[0] != 0:
        sub_divided_sents = [all_sentences[0: section[0]]] + sub_divided_sents

    for sentence1 in sub_divided_sents:
        print('sub divided: ', sentence1, "\n\n")

    # sub divided is good: purpose, identifies short sentences in the text and flags them maybe as headings.

    # merge all indicies with a single element with the next
    return combineSentences(sub_divided_sents)



def combineChunks(chunksArr):
     # merge all indicies with a single element with the next
    # print("len chunks array pre: ", len(chunksArr))

    to_delete = []
    totalSentences = len(chunksArr)
    for i in range(1, totalSentences):
        prev_division = chunksArr[i-1]
        curr_division = chunksArr[i]
        prev_division_len = len(prev_division)

        # print("previous division len: ", prev_division_len)
        if (prev_division_len < 128):
            # print('add previoud chunk: ', prev_division, 'to ', curr_division[0: 70])
            chunksArr[i] = prev_division + curr_division
            to_delete.append(i-1)


    # now check if the last index is too short. Combine it with previous
    if len(chunksArr[-1]) < 128 and totalSentences > 1:
        chunksArr[-2] = chunksArr[-2] + chunksArr[-1]
        to_delete.append(-1)

    # print("len chunks array: ", len(chunksArr))

    # for chunk in chunksArr:
    #     print(chunk)
    #     print("\n\n")

    # print ('to deltee: ', to_delete)

    # delete all merged indicies
    for i in range(len(to_delete)-1, -1, -1): 
        del chunksArr[to_delete[i]]

    # print("len chunks array after del: ", len(chunksArr))

    return chunksArr

# break it down into as many pieces
def lower_chunk_text(text, threshold):
    sents_div, vecs_div = process(text)
    reclusters = cluster_text(sents_div, vecs_div, threshold)
    final_texts = []
            
    for subcluster in reclusters:
        div_txt = clean_text(' '.join([sents_div[i].text for i in subcluster]))
        div_len = len(div_txt)
        
        if div_len > upper_limit:
            return lower_chunk_text(text, threshold=threshold+0.1)
        else: final_texts.append(div_txt)

    return final_texts



def empty_buffer(final_chunks, buffer):
    if (len(buffer)> 0):
        final_chunks.append(buffer)
        buffer=""
    return final_chunks, buffer

def empty_buffer_if_needed(final_chunks, buffer, last_chunk):
    if (len(buffer)> 0):
        final_chunks.append(buffer)
        buffer=last_chunk
    return final_chunks, buffer


def empty_buffer_if_needed_1(final_chunks, buffer, final_texts, last_chunk):
    if (len(final_texts) > 1):
        # print("EMPTYING BUFFER: ", buffer, "\n\n")

        final_chunks.append(buffer)
        buffer=last_chunk
    else:
        buffer = final_texts[0]
    return final_chunks, buffer

# the initial method is to lower the overall length of a chunk. Some chunks may become too small. So reshuffle.
def recluster_small_chunks(chunks):
    final_chunks=[]
    buffer = chunks[0] # a combined string of chunks
    for i in range(1, len(chunks)):
        combined = buffer+chunks[i]
        if (len(combined) < upper_limit):
            final_texts = lower_chunk_text(combined, 0.4)
            final_chunks, buffer = empty_buffer_if_needed_1(final_chunks, buffer, final_texts, chunks[i])
        else:
            final_chunks, buffer = empty_buffer(final_chunks, buffer)
            buffer=chunks[i]
            
    final_chunks, buffer = empty_buffer(final_chunks, buffer)

    return final_chunks

def chunk_text(text):
    # Initialize the clusters lengths list and final texts list

    final_texts = []

    # Process the chunk
    threshold = 0.9

    combined  = run_manual_chunk(text)
    texts = []
    # now for each combined chunk, run semantic chunking.
    final_chunks = []
    # final_texts = lower_chunk_text(text, threshold)
    # final_chunks += final_texts

    print("combined len: ", len(combined))
   
    for i in range (0, len(combined)):
        final_texts = lower_chunk_text(combined[i], threshold)
        final_chunks += combineChunks(final_texts)
        # final_chunks += final_texts
    
    
    
    # print('CLUSTERS:\n\n')
    # print(reclusters)
    print('\n\n')

    # new_chunks = recluster_small_chunks(final_texts)

    return final_chunks