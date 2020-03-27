import tensorflow as tf

from DATA import *
from CONVNET import *

def conv_net_block(conv_net, net_info, tensor_list, is_first, is_training, act_o):
    seed = FLAGS['process_random_seed']
    trainable = conv_net['trainable']
    tensor = tensor_list[conv_net['input_index']]
    if is_first:
        layer_name_format = '%12s'
        net_info.architecture_log.append('========== net_name = %s ==========' % conv_net['net_name'])
        net_info.architecture_log.append('[%s][%4d] : (%s)' % (layer_name_format % 'input', tensor_list.index(tensor), ', '.join('%4d' % (-1 if v is None else v) for v in tensor.get_shape().as_list())))
        if FLAGS['mode_use_debug']:
            print(net_info.architecture_log[-2])
            print(net_info.architecture_log[-1])
    with tf.compat.v1.variable_scope(conv_net['net_name']):
        for l_index, layer_o in enumerate(conv_net['layers']):
            layer = layer_o['name']
            #this should be cheanged to an enum or dict mapping
            if layer == "relu":
                tensor = exe_relu_layer(tensor)
            elif layer == "prelu":
                tensor = exe_prelu_layer(tensor, net_info, l_index, is_first, act_o)
            elif layer == "lrelu":
                tensor = exe_lrelu_layer(tensor, layer_o)
            elif layer == "bn":
                tensor = exe_bn_layer(tensor, layer_o, net_info, l_index, is_first, is_training, trainable, act_o)
            elif layer == "in":
                tensor = exe_in_layer(tensor, layer_o, net_info, l_index, is_first, trainable, act_o)
            elif layer == "ln":
                tensor = exe_ln_layer(tensor, layer_o, net_info, l_index, is_first, trainable, act_o)
            elif layer == "conv":
                tensor = exe_conv_layer(tensor, layer_o, net_info, l_index, is_first, is_training, trainable, seed)
            elif layer == "conv_res":
                tensor = exe_conv_res_layer(tensor, layer_o, tensor_list, net_info, l_index, is_first, is_training, trainable, seed)
            elif layer == "res":
                tensor = exe_res_layer(tensor, layer_o, tensor_list)
            elif layer == "max_pool":
                tensor = exe_max_pool_layer(tensor, layer_o)
            elif layer == "avg_pool":
                tensor = exe_avg_pool_layer(tensor, layer_o)
            elif layer == "resize":
                tensor = exe_resize_layer(tensor, layer_o)
            elif layer == "concat":
                tensor = exe_concat_layer(tensor, layer_o, tensor_list)
            elif layer == "g_concat":
                tensor = exe_global_concat_layer(tensor, layer_o, tensor_list)
            elif layer == "reshape":
                tensor = exe_reshape_layer(tensor, layer_o)
            elif layer == "clip":
                tensor = exe_clip_layer(tensor, layer_o)
            elif layer == "sigmoid":
                tensor = exe_sigmoid_layer(tensor)
            elif layer == "softmax":
                tensor = exe_softmax_layer(tensor)
            elif layer == "squeeze":
                tensor = exe_squeeze_layer(tensor, layer_o)
            elif layer == "abs":
                tensor = exe_abs_layer(tensor)
            elif layer == "tanh":
                tensor = exe_tanh_layer(tensor)
            elif layer == "inv_tanh":
                tensor = exe_inv_tanh_layer(tensor)
            elif layer == "add":
                tensor = exe_add_layer(tensor, layer_o)
            elif layer == "mul":
                tensor = exe_mul_layer(tensor, layer_o)
            elif layer == "reduce_mean":
                tensor = exe_reduce_mean_layer(tensor, layer_o)
            elif layer == "null":
                tensor = exe_null_layer(tensor)
            elif layer == "selu":
                tensor = exe_selu_layer(tensor)
            else:
                assert False, 'Error layer name = %s' % layer
            tensor_list.append(tensor)

            if is_first:
                info = '[%s][%4d] : (%s)'% (layer_name_format % layer, tensor_list.index(tensor), ', '.join('%4d' % (-1 if v is None else v) for v in tensor.get_shape().as_list()))
                if 'index' in layer_o:
                    info = info + ', use index [%4d] : (%s)' % (layer_o['index'], ', '.join('%4d' % (-1 if v is None else v) for v in tensor_list[layer_o['index']].get_shape().as_list()))
                net_info.architecture_log.append(info)
                if FLAGS['mode_use_debug']:
                    print(info)

    return tensor

def model(net_info, tensor, is_training, act_o, is_first=False):
    tensor_list = [tensor]
    if net_info.name == "netD":
        for net_n in net_info.CONV_NETS:
            _ = conv_net_block(net_n, net_info, tensor_list, is_first, is_training, act_o)
        result = tensor_list[-1]
    elif net_info.name == "netG":
        for net_n in net_info.CONV_NETS:
            _ = conv_net_block(net_n, net_info, tensor_list, is_first, is_training, act_o)
        result = tensor_list[-1]
    else:
        assert False, 'net_info.name ERROR = %s' % net_info.name
    return result

def img_L2_loss(img1, img2, use_local_weight):
    if use_local_weight:
        w = -tf.math.log(tf.cast(img2, tf.float64) + tf.exp(tf.constant(-99, dtype=tf.float64))) + 1
        w = tf.cast(w * w, tf.float32)
        return tf.reduce_mean(input_tensor=w * tf.square(tf.subtract(img1, img2)))
    else:
        return tf.reduce_mean(input_tensor=tf.square(tf.subtract(img1, img2)))

def img_L1_loss(img1, img2):
    return tf.reduce_mean(input_tensor=tf.abs(tf.subtract(img1, img2)))

def img_GD_loss(img1, img2):
    img1 = tf_imgradient(tf.pack([img1]))
    img2 = tf_imgradient(tf.pack([img2]))
    return tf.reduce_mean(input_tensor=tf.square(tf.subtract(img1, img2)))

def regularization_cost(net_info):
    cost = 0
    for w, p in zip(net_info.weights, net_info.parameter_names):
        if p[-2:] == "_w": 
            cost = cost + (tf.nn.l2_loss(w))
    return cost


def initialize_model(FLAGS):

def netG_concat_value(tensor, v):
    v_t = tf.constant(v, dtype=tf.float32, shape=tensor.get_shape().as_list()[:3] + [1])
    tensor = tf.concat(3, [tensor, v_t])
    return tensor
netG_act_o_1 = dict(size=2, index=0)
netG_act_o_2 = dict(size=2, index=1)
netD_act_o   = dict(size=1, index=0)


def get_netG_outputs(netG,train_df,test_df, FLAGS)
    with tf.compat.v1.name_scope(netG.name):
        with tf.compat.v1.variable_scope(netG.variable_scope_name) as scope_full:
            with tf.compat.v1.variable_scope(netG.variable_scope_name + 'B') as scopeB:
                netG_train_output2 = model(netG, train_df.input2, True, netG_act_o_1, is_first=True)
                scopeB.reuse_variables()
                netG_test_output2  = model(netG, test_df.input2, False, netG_act_o_1)
                netG_train_output2_for_netD = model(netG, train_df.input2, False, netG_act_o_1)

            with tf.compat.v1.variable_scope(netG.variable_scope_name + 'A') as scopeA:
                netG_train_output1 = model(netG, train_df.input1, True, netG_act_o_1, is_first=True)
                scopeA.reuse_variables()
                netG_test_output1  = model(netG, test_df.input1, False, netG_act_o_1)
                netG_train_output1_for_netD = model(netG, train_df.input1, False, netG_act_o_1)
                netG_train_output2_inv = model(netG, tf.clip_by_value(netG_train_output2, 0, 1),  True, netG_act_o_2)
                netG_test_output2_inv  = model(netG, tf.clip_by_value(netG_test_output2,  0, 1), False, netG_act_o_2)

            with tf.compat.v1.variable_scope(netG.variable_scope_name + 'B') as scopeB:
                scopeB.reuse_variables()
                netG_train_output1_inv = model(netG, tf.clip_by_value(netG_train_output1, 0, 1),  True, netG_act_o_2)
                netG_test_output1_inv  = model(netG, tf.clip_by_value(netG_test_output1,  0, 1), False, netG_act_o_2)
    
    netG_train_outputs = [netG_train_output1 , netG_train_output2]
    netG_test_outputs = [netG_test_output1 , netG_test_output2]
    netG_train_output_for_netD_list = [netG_train_output1_for_netD , netG_train_output2_for_netD]
    netG_train_output_inv_list = [netG_train_output1_inv , netG_train_output2_inv]
    netG_test_output_inv_list = [netG_test_output1_inv , netG_test_output2_inv]

    return netG_train_outputs , netG_test_outputs , netG_train_output_for_netD_list , netG_train_output_inv_list ,netG_test_output_inv_list

def wgan_gp(fake_data, real_data):
    fake_data = tf.reshape(fake_data, [FLAGS['data_train_batch_size'], -1])
    real_data = tf.reshape(real_data, [FLAGS['data_train_batch_size'], -1])
    alpha = tf.random.uniform(shape=[FLAGS['data_train_batch_size'], 1], minval=0., maxval=1., seed=FLAGS['process_random_seed'])
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    interpolates_D = tf.reshape(interpolates, [FLAGS['data_train_batch_size'], FLAGS['data_image_size'], FLAGS['data_image_size'], FLAGS['data_image_channel']])
    gradients = tf.gradients(ys=model(netD, interpolates_D, True, netD_act_o), xs=[interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(input_tensor=tf.square(gradients), axis=[1]))
    if FLAGS['loss_wgan_use_g_to_one']:
        gradient_penalty = -tf.reduce_mean(input_tensor=(slopes-1.)**2)
    else:
        gradient_penalty = -tf.reduce_mean(input_tensor=tf.maximum(0., slopes-1.))
    return gradient_penalty

def get_D_G_outputs(netD,netG,FLAGS):
    with tf.compat.v1.name_scope(netD.name):
        with tf.compat.v1.variable_scope(netD.variable_scope_name) as scope_full:
            with tf.compat.v1.variable_scope(netD.variable_scope_name + 'A') as scopeA:
                netD_train_output1_1 = model(netD, netG_train_output1_for_netD, True, netD_act_o, is_first=True)
                scopeA.reuse_variables()
                netD_train_output2_1 = model(netD, train_df.input2, True, netD_act_o)
                netD_netG_train_output1_1 = model(netD, netG_train_output1, True, netD_act_o)
                netD_netG_train_output2_1 = netD_train_output2_1
                netD_test_output1_1 = model(netD, netG_test_output1, False, netD_act_o)
                netD_test_output2_1 = model(netD, test_df.input2, False, netD_act_o)
                # wgan-gp
                if FLAGS['loss_wgan_gp_use_all']:
                    assert False, 'not yet'
                    gradient_penalty = tf.reduce_mean(input_tensor=tf.stack([(\
                        wgan_gp(netD_train_input1, netD_train_input2) + wgan_gp(train_df.input1, netD_train_input1) + wgan_gp(train_df.input1, netD_train_input2)) / 3.0 \
                    for _ in range(FLAGS['loss_wgan_gp_times'])]))
                else:
                    w_list = []
                    for _ in range(FLAGS['loss_wgan_gp_times']):
                        w_list.append(wgan_gp(netG_train_output1_for_netD, train_df.input2))
                    gradient_penalty_1 = tf.reduce_mean(input_tensor=tf.stack(w_list)) * gp_weight_1
            with tf.compat.v1.variable_scope(netD.variable_scope_name + 'B') as scopeB:
                netD_train_output1_2 = model(netD, train_df.input1, True, netD_act_o, is_first=True)
                scopeB.reuse_variables()
                netD_train_output2_2 = model(netD, netG_train_output2_for_netD, True, netD_act_o)
                netD_netG_train_output1_2 = netD_train_output1_2
                netD_netG_train_output2_2 = model(netD, netG_train_output2, True, netD_act_o)
                netD_test_output1_2 = model(netD, test_df.input1, False, netD_act_o)
                netD_test_output2_2 = model(netD, netG_test_output2, False, netD_act_o)
                # wgan-gp
                if FLAGS['loss_wgan_gp_use_all']:
                    assert False, 'not yet'
                    gradient_penalty = tf.reduce_mean(input_tensor=tf.stack([(\
                        wgan_gp(netD_train_input1, netD_train_input2) + wgan_gp(train_df.input1, netD_train_input1) + wgan_gp(train_df.input1, netD_train_input2)) / 3.0 \
                    for _ in range(FLAGS['loss_wgan_gp_times'])]))
                else:
                    w_list = []
                    for _ in range(FLAGS['loss_wgan_gp_times']):
                        w_list.append(wgan_gp(netG_train_output2_for_netD, train_df.input1))
                    gradient_penalty_2 = tf.reduce_mean(input_tensor=tf.stack(w_list)) * gp_weight_2
    
    netD_train_outputs = [netD_train_output1_1,netD_train_output2_1,netD_train_output1_2,netD_train_output2_2]
    netD_netG_train_outputs = [netD_netG_train_output1_1,netD_netG_train_output2_1,,netD_netG_train_output1_2,netD_netG_train_output2_2]
    gradient_penalties = [gradient_penalty_1,gradient_penalty_2]

    return  netD_train_outputs,netD_netG_train_outputs, gradient_penalties

def get_data_terms(netG_crops):
     if FLAGS['loss_source_data_term_weight'] > 0:
            if FLAGS['loss_source_data_term'] == 'l2':
                train_data_term_1 = -tf.reduce_mean(input_tensor=tf.stack([img_L2_loss(a, b, FLAGS['loss_data_term_use_local_weight']) for a, b in zip(netG_train_output1_crop, netG_train_input1_crop)])) * FLAGS['loss_source_data_term_weight']
                test_data_term_1  = -img_L2_loss(netG_test_output1_crop, netG_test_input1_crop, FLAGS['loss_data_term_use_local_weight']) * FLAGS['loss_source_data_term_weight']
                train_data_term_2 = -tf.reduce_mean(input_tensor=tf.stack([img_L2_loss(a, b, FLAGS['loss_data_term_use_local_weight']) for a, b in zip(netG_train_output2_crop, netG_train_input2_crop)])) * FLAGS['loss_source_data_term_weight']
                test_data_term_2  = -img_L2_loss(netG_test_output2_crop, netG_test_input2_crop, FLAGS['loss_data_term_use_local_weight']) * FLAGS['loss_source_data_term_weight']
            elif FLAGS['loss_source_data_term'] == 'l1':
                assert False, 'not yet'
                train_data_term_1 = -tf.reduce_mean(input_tensor=tf.stack([img_L1_loss(a, b) for a, b in zip(netG_train_output1_crop, netG_train_input1_crop)])) * FLAGS['loss_source_data_term_weight']
                test_data_term_1  = -img_L1_loss(netG_test_output1_crop, netG_test_input1_crop) * FLAGS['loss_source_data_term_weight']
                train_data_term_2 = -tf.reduce_mean(input_tensor=tf.stack([img_L1_loss(a, b) for a, b in zip(netG_train_output2_crop, netG_train_input2_crop)])) * FLAGS['loss_source_data_term_weight']
                test_data_term_2  = -img_L1_loss(netG_test_output2_crop, netG_test_input2_crop) * FLAGS['loss_source_data_term_weight']
            elif FLAGS['loss_source_data_term'] == 'PR':
                assert False, 'not yet'
                train_data_term_1 =  tf.stack([tf_photorealism_loss(netG_train_output1, train_df.mat1, i, FLAGS['loss_photorealism_is_our']) for i in range(FLAGS['data_train_batch_size'])])
                train_data_term_1 = -tf.reduce_mean(input_tensor=train_data_term_1) * FLAGS['loss_source_data_term_weight']
                test_data_term_1  =  tf.stack([tf_photorealism_loss(netG_test_output1,  test_df.mat1,  0, FLAGS['loss_photorealism_is_our'])])
                test_data_term_1  = -tf.reduce_mean(input_tensor=test_data_term_1)  * FLAGS['loss_source_data_term_weight']
            else:
                assert False, 'data term error = %s' % FLAGS['loss_source_data_term']
        else:
            train_data_term_1 = tf.constant(0, dtype=FLAGS['data_compute_dtype'])
            test_data_term_1 = tf.constant(0, dtype=FLAGS['data_compute_dtype'])
            train_data_term_2 = tf.constant(0, dtype=FLAGS['data_compute_dtype'])
            test_data_term_2 = tf.constant(0, dtype=FLAGS['data_compute_dtype'])

    train_data_terms= [train_data_term_1 ,train_data_term_2 ]
    test_data_terms= [test_data_term_1 , test_data_term_2 ]

    return train_data_terms , test_data_terms 

def get_netD_loss(netG_crops):
     if FLAGS['loss_constant_term_weight'] > 0:
            netG_train_output1_inv_crop = [tf_crop_rect(netG_train_output1_inv, train_df.mat1, i) for i in range(FLAGS['data_train_batch_size'])]
            netG_test_output1_inv_crop  =  tf_crop_rect(netG_test_output1_inv,  test_df.mat1,  0)
            netG_train_output2_inv_crop = [tf_crop_rect(netG_train_output2_inv, train_df.mat2, i) for i in range(FLAGS['data_train_batch_size'])]
            netG_test_output2_inv_crop  =  tf_crop_rect(netG_test_output2_inv,  test_df.mat2,  0)
            if FLAGS['loss_constant_term'] == 'l2':
                train_constant_term_1 = -tf.reduce_mean(input_tensor=tf.stack([img_L2_loss(a, b, FLAGS['loss_constant_term_use_local_weight']) for a, b in zip(netG_train_output1_inv_crop, netG_train_input1_crop)])) * FLAGS['loss_constant_term_weight']
                test_constant_term_1  = -img_L2_loss(netG_test_output1_inv_crop, netG_test_input1_crop, FLAGS['loss_constant_term_use_local_weight']) * FLAGS['loss_constant_term_weight']
                train_constant_term_2 = -tf.reduce_mean(input_tensor=tf.stack([img_L2_loss(a, b, FLAGS['loss_constant_term_use_local_weight']) for a, b in zip(netG_train_output2_inv_crop, netG_train_input2_crop)])) * FLAGS['loss_constant_term_weight']
                test_constant_term_2  = -img_L2_loss(netG_test_output2_inv_crop, netG_test_input2_crop, FLAGS['loss_constant_term_use_local_weight']) * FLAGS['loss_constant_term_weight']
            elif FLAGS['loss_constant_term'] == 'l1':
                train_constant_term_1 = -tf.reduce_mean(input_tensor=tf.stack([img_L1_loss(a, b) for a, b in zip(netG_train_output1_inv_crop, netG_train_input1_crop)])) * FLAGS['loss_constant_term_weight']
                test_constant_term_1  = -img_L1_loss(netG_test_output1_inv_crop, netG_test_input1_crop) * FLAGS['loss_constant_term_weight']
                train_constant_term_2 = -tf.reduce_mean(input_tensor=tf.stack([img_L1_loss(a, b) for a, b in zip(netG_train_output2_inv_crop, netG_train_input2_crop)])) * FLAGS['loss_constant_term_weight']
                test_constant_term_2  = -img_L1_loss(netG_test_output2_inv_crop, netG_test_input2_crop) * FLAGS['loss_constant_term_weight']
            elif FLAGS['loss_constant_term'] == 'PR':
                assert False, 'not yet'
                train_constant_term_1 =  tf.stack([tf_photorealism_loss(netG_train_output1_inv, train_df.mat1, i, FLAGS['loss_photorealism_is_our']) for i in range(FLAGS['data_train_batch_size'])])
                train_constant_term_1 = -tf.reduce_mean(input_tensor=train_constant_term_1) * FLAGS['loss_constant_term_weight']
                test_constant_term_1  =  tf.stack([tf_photorealism_loss(netG_test_output1_inv,  test_df.mat1,  0, FLAGS['loss_photorealism_is_our'])])
                test_constant_term_1  = -tf.reduce_mean(input_tensor=test_constant_term_1)  * FLAGS['loss_constant_term_weight']
            else:
                assert False, 'constant data term error = %s' % FLAGS['loss_constant_term']

        else:
            train_constant_term_1 = tf.constant(0, dtype=FLAGS['data_compute_dtype'])
            test_constant_term_1 = tf.constant(0, dtype=FLAGS['data_compute_dtype'])
            train_constant_term_2 = tf.constant(0, dtype=FLAGS['data_compute_dtype'])
            test_constant_term_2 = tf.constant(0, dtype=FLAGS['data_compute_dtype'])

        netD_train_loss = (-tf.reduce_mean(input_tensor=netD_train_output1_1) + tf.reduce_mean(input_tensor=netD_train_output2_1)) + (-tf.reduce_mean(input_tensor=netD_train_output1_2) + tf.reduce_mean(input_tensor=netD_train_output2_2))
        netD_test_loss  = (-tf.reduce_mean(input_tensor=netD_test_output1_1)  + tf.reduce_mean(input_tensor=netD_test_output2_1))  + (-tf.reduce_mean(input_tensor=netD_test_output1_2)  + tf.reduce_mean(input_tensor=netD_test_output2_2))
    
    train_constant_term_list = [train_constant_term_1 , train_constant_term_2] 
    test_constant_term_list = [test_constant_term_1 , test_constant_term_2]


    return netD_train_loss, netD_test_loss , train_constant_term_list ,test_constant_term_list

def get_net_G_loss(netD_netG_train_outputs,netD_netG_test_outputs,netD_test_outputs):


     def netG_improve_loss(be, af):
            l = af - be
            l = tf.reduce_mean(input_tensor=tf.sign(l) * tf.square(l))
            return tf.sign(l) * tf.sqrt(tf.abs(l))

    netG_train_loss = tf.reduce_mean(input_tensor=netD_netG_train_output1_1) - tf.reduce_mean(input_tensor=netD_netG_train_output2_2)
    netG_test_loss  = tf.reduce_mean(input_tensor=netD_test_output1_1)       - tf.reduce_mean(input_tensor=netD_test_output2_2)
    netG_batch_list_train_loss = netD_netG_train_output1_1 - netD_netG_train_output2_2

    netG_train_1_1 = tf.reduce_mean(input_tensor=netD_netG_train_output1_1)
    netG_train_2_1 = tf.reduce_mean(input_tensor=netD_netG_train_output2_1)
    netG_train_1_2 = tf.reduce_mean(input_tensor=netD_netG_train_output1_2)
    netG_train_2_2 = tf.reduce_mean(input_tensor=netD_netG_train_output2_2)

    netD_train_1_1 = tf.reduce_mean(input_tensor=netD_train_output1_1)
    netD_train_2_1 = tf.reduce_mean(input_tensor=netD_train_output2_1)
    netD_train_1_2 = tf.reduce_mean(input_tensor=netD_train_output1_2)
    netD_train_2_2 = tf.reduce_mean(input_tensor=netD_train_output2_2)

    netG_test_1_1   = tf.reduce_mean(input_tensor=netD_test_output1_1)
    netG_test_2_1   = tf.reduce_mean(input_tensor=netD_test_output2_1)
    netG_test_1_2   = tf.reduce_mean(input_tensor=netD_test_output1_2)
    netG_test_2_2   = tf.reduce_mean(input_tensor=netD_test_output2_2)

    netG_train_list = [netG_train_1_1 ,netG_train_2_1,netG_train_1_2 ,netG_train_2_2 ]
    netD_train_list = [netD_train_1_1 ,netD_train_2_1,netD_train_1_2 ,netD_train_2_2]
    netG_test_list = [netG_test_1_1 ,netG_test_2_1,netG_test_1_2 ,netG_test_2_2]

    return netG_train_loss, netG_test_loss ,netG_batch_list_train_loss , netG_train_list, netD_train_list, netG_test_list


def get_losses(net_G_train_outputs , net_G_test_outputs, train_df, test_df ):
    with tf.compat.v1.name_scope("Loss"):

        netG_train_output1_crop = [tf_crop_rect(netG_train_output1, train_df.mat1, i) for i in range(FLAGS['data_train_batch_size'])]
        netG_train_output2_crop = [tf_crop_rect(netG_train_output2, train_df.mat2, i) for i in range(FLAGS['data_train_batch_size'])]
        netG_train_input1_crop  = [tf_crop_rect(train_df.input1,    train_df.mat1, i) for i in range(FLAGS['data_train_batch_size'])]
        netG_train_input2_crop  = [tf_crop_rect(train_df.input2,    train_df.mat2, i) for i in range(FLAGS['data_train_batch_size'])]

        netG_train_input1_label_crop  = [tf_crop_rect(train_df.input1_label, train_df.mat1, i) for i in range(FLAGS['data_train_batch_size'])]
        netG_train_input2_label_crop  = [tf_crop_rect(train_df.input2_label, train_df.mat2, i) for i in range(FLAGS['data_train_batch_size'])]

        netG_test_output1_crop  =  tf_crop_rect(netG_test_output1, test_df.mat1, 0)
        netG_test_output2_crop  =  tf_crop_rect(netG_test_output2, test_df.mat2, 0)
        netG_test_input1_crop   =  tf_crop_rect(test_df.input1,    test_df.mat1, 0)
        netG_test_input2_crop   =  tf_crop_rect(test_df.input2,    test_df.mat2, 0)

        net_G_crops=[netG_train_output1_crop,netG_train_output2_crop,netG_train_input1_crop,netG_train_input2_crop\
            , netG_train_input1_label_crop,netG_train_input2_label_crop,netG_test_output1_crop,netG_test_output2_crop,netG_test_input1_crop,netG_test_input2_crop  ]

        train_data_term_1 ,test_data_term_1 ,  train_data_term_2 ,test_data_term_2= get_data_terms()

        get_netD_loss()
        get_netG_loss()
       

        netG_loss = netG_train_loss + train_data_term_1 + train_data_term_2 + train_constant_term_1 + train_constant_term_2
        netD_loss = netD_train_loss + gradient_penalty_1 + gradient_penalty_2

    netG_total_loss = -netG_loss + netG_w_regularization_loss
    netD_total_loss = -netD_loss + netD_w_regularization_loss
    netG_train_summary = [netG_train_loss, netG_train_1_1, netG_train_2_1, netG_train_1_2, netG_train_2_2, train_data_term_1, train_data_term_2, train_constant_term_1, train_constant_term_2, netG_tr_psnr1, netG_tr_psnr2, netG_r_loss, netG_gbc, netG_gac]
    return netG_total_loss, netD_total_loss, netG_train_list,netD_train_list,netG_test_list,train_data_term_lit, train_constant_term_list

def get_G_gradient():
    with tf.compat.v1.name_scope("netG_SGD"):
        netG_optimizer = netG.OPTIMIZER
        netG_gvs = netG_optimizer.compute_gradients(netG_total_loss, tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=netG.variable_scope_name))
        netG_gbc = [grad for grad, var in netG_gvs]
        netG_capped_gvs = [(tf.clip_by_value(grad, -netG.GLOBAL_GRADIENT_CLIPPING, netG.GLOBAL_GRADIENT_CLIPPING), var) for grad, var in netG_gvs]
        netG_gac = [grad for grad, var in netG_capped_gvs]
        netG_opt = netG_optimizer.apply_gradients(netG_capped_gvs)

        netG_gbc = tf.reduce_mean(input_tensor=tf.stack([tf.reduce_mean(input_tensor=tf.abs(v)) for v in netG_gbc]))
        netG_gac = tf.reduce_mean(input_tensor=tf.stack([tf.reduce_mean(input_tensor=tf.abs(v)) for v in netG_gac]))
    return netG_opt,netG_gbc, netG_gac

def get_D_gradient():
    with tf.compat.v1.name_scope("netD_SGD"):
        netD_optimizer = netD.OPTIMIZER
        netD_gvs = netD_optimizer.compute_gradients(netD_total_loss, tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=netD.variable_scope_name))
        netD_gbc = [grad for grad, var in netD_gvs]
        netD_capped_gvs = [(tf.clip_by_value(grad, -netD.GLOBAL_GRADIENT_CLIPPING, netD.GLOBAL_GRADIENT_CLIPPING), var) for grad, var in netD_gvs]
        netD_gac = [grad for grad, var in netD_capped_gvs]
        netD_opt = netD_optimizer.apply_gradients(netD_capped_gvs)

        netD_gbc = tf.reduce_mean(input_tensor=tf.stack([tf.reduce_mean(input_tensor=tf.abs(v)) for v in netD_gbc]))
        netD_gac = tf.reduce_mean(input_tensor=tf.stack([tf.reduce_mean(input_tensor=tf.abs(v)) for v in netD_gac]))
    r
    eturn netD_opt,netD_gbc, netD_gac

















