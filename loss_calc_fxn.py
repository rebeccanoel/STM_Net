'''
Loss Calculation Function
'''

import tensorflow as tf

def loss_calc(logits, labels):
    
    llist = []
    #for n in range(len(labels)):
    llist.append(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits[-1],labels=labels[0])))
    #final_loss = llist[0]
    loss = llist[0]
    for r in range(1, len(llist)):
        loss += llist[r]
    final_loss = loss#/len(llist)
    return final_loss, llist