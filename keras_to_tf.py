from keras.models import load_model
from keras import backend as K
import tensorflow as tf

keras_model = "cnn/models/hand_poses_wGarbage_10.h5"

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.
    
    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                            or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                        output_names, freeze_var_names)
        return frozen_graph


# loading keras model
K.set_learning_phase(0)
model = load_model(keras_model)

# create a frozen-graph of the keras model
frozen_graph = freeze_session(K.get_session(),
                              output_names=[out.op.name for out in model.outputs])

# save model as .pb file
tf.train.write_graph(frozen_graph, "cnn/models/TF/", "tf_model.pb", as_text=False)