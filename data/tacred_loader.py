import json


class TACREDExample(object):
    """ A single train/test exmaple for the TAC RED dataset. """

    def __init__(self, 
                ex_id, # unique id for each example
                context, # text 
                subj, # subj position list [st, ed]
                obj, # obj position list [st, ed]
                subj_type, # entity type of subj
                obj_type, # entity type of obj
                pos, # stanford pos 
                ner, # stanford ner
                head, # stanford dependency tree head
                deprel, # stanford dependency tree edge relation 
                label):
        self.ex_id = id
        self.context = context
        self.subj = subj
        self.subj_type = subj_type
        self.obj = obj
        self.obj_type = obj_type
        self.pos = pos
        self.ner = ner
        self.head = head
        self.deprel = deprel
        self.label = label
    
class InputFeature(object):
    """ A single set of features of data. """

    def __init__(self,
                input_ids,
                input_mask,
                segment_ids,
                label_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segmemt_ids = segment_ids
        self.label_ids = label_ids
        


    

