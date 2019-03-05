import tensorflow as tf
file = './pb_out_new.pb'
with tf.gfile.FastGFile("./pb_out_new.pb", 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')

    for node in graph_def.node:
    	print(node.input)



print('Graph loaded.')