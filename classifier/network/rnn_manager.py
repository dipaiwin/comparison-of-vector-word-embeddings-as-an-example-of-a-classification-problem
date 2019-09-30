import tensorflow as tf
from network.values_manager import Values
from config_project import ConfigProject


class RnnManager:
    def __init__(self, config: ConfigProject, values: Values):
        self.cfg = config
        self.values = values
        self.cells_fw, self.cells_fw_do = self.create_cell()
        self.cell_bw, self.cell_bw_do = self.create_cell()
    
    def create_cell(self):
        c = []
        c_w = []
        for _ in range(self.cfg['count_rnn_layers']):
            cell = tf.contrib.rnn.GRUCell(self.cfg['output_rnn'])
            c.append(cell)
            
            cell_wrapper = tf.nn.rnn_cell.DropoutWrapper(
                cell,
                self.cfg['do_cell_input'],
                self.cfg['do_cell_hidden']
            )
            c_w.append(cell_wrapper)
        
        multi_cell = tf.contrib.rnn.MultiRNNCell(c)
        multi_cell_do = tf.contrib.rnn.MultiRNNCell(c_w)
        
        return multi_cell, multi_cell_do
    
    def run_rnn(self, inputs):
        return tf.cond(
            self.values['is_training'],
            lambda: self.run_rnn_start(inputs, self.cells_fw_do, self.cell_bw_do),
            lambda: self.run_rnn_start(inputs, self.cells_fw, self.cell_bw)
        )
    
    def run_rnn_start(self, inputs, cells_fw, cells_bw):
        # [B x L(w) x C]
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
            cells_fw,
            cells_bw,
            inputs,
            dtype=tf.float32,
            sequence_length=self.values['lengths']
        )
        
        # # [B, N(RNN layers), 2C]
        output_state = tf.concat(
            [
                # [B, N(RNN layers), C]
                tf.stack(output_states[0], 1),
                # [B, N(RNN layers), C]
                tf.stack(output_states[1], 1)
            ],
            2
        )
        outputs = tf.concat(outputs, 2)
        # return outputs, output_state
        return output_state
