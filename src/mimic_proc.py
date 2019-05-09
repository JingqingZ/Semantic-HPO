
import config, dataloader

def get_sentences_from_mimic(text):
    section_title = [
        'history of present illness',
        'past medical history',
        'social history',
        'family history',
        'physical exam',
        'pertinent results',
        'discharge labs',
        'radiology',
        'microbiology',
        'brief hospital course'
    ]

    loc_list = []
    for sec in section_title:
        loc = text.find(sec)
        if loc != -1:
            loc_list.append(loc)

    sentences = text.split("\n")
    new_sentences = list()
    cur_loc = 0

    start_loc = loc_list[-1] if len(loc_list) > 0 else 0
    for s in sentences:
        cur_loc += len(s)
        if cur_loc > start_loc and len(s) > 0:
            new_sentences.append(s)

    return new_sentences


if __name__ == '__main__':
    mimic_corpus, _ = dataloader.get_corpus()
    for i in range(len(mimic_corpus)):
        get_sentences_from_mimic(mimic_corpus[i])
    pass
