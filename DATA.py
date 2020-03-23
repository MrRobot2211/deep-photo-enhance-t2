import tensorflow as tf
import numpy as np
import random, cv2, operator, os

from multiprocessing import Process, Queue, Manager
from FUNCTION import *
from CONVNET import *

tf.compat.v1.disable_eager_execution()
tf.compat.v1.disable_resource_variables()
# Configure
FLAGS = {}

FLAGS['method'] = 'WGAN-v24-cycleganD2' 
FLAGS['mode_use_debug'] = False
FLAGS['num_exp'] = 736
FLAGS['num_gpu'] = '3'
FLAGS['sys_use_unix'] = True
FLAGS['sys_is_dgx'] = True

FLAGS['netD_init_method'] = 'var_scale' #var_scale, rand_uniform, rand_normal, truncated_normal
FLAGS['netD_init_weight'] = 1e-3
FLAGS['netD_base_learning_rate'] = 1e-5
FLAGS['netD_base_learning_decay'] = 75
FLAGS['netD_base_learning_decay_epoch'] = 75
FLAGS['netD_regularization_weight'] = 0
FLAGS['netD_times'] = 50
FLAGS['netD_times_grow'] = 1
FLAGS['netD_buffer_times'] = 50 #it depends on batch size
FLAGS['netD_init_times'] = 0
FLAGS['netG_init_method'] = 'var_scale' #var_scale, rand_uniform, rand_normal, truncated_normal
FLAGS['netG_init_weight'] = 1e-3
FLAGS['netG_base_learning_rate'] = 1e-5
FLAGS['netG_base_learning_decay'] = 75
FLAGS['netG_base_learning_decay_epoch'] = 75
FLAGS['netG_regularization_weight'] = 0
FLAGS['loss_source_data_term'] = 'l2' # l1, l2, PR, GD
FLAGS['loss_source_data_term_weight'] = 1e3
FLAGS['loss_constant_term'] = 'l2' # l1, l2, PR, GD
FLAGS['loss_constant_term_weight'] = 1e4
FLAGS['loss_photorealism_is_our'] =  True
FLAGS['loss_wgan_lambda'] = 10
FLAGS['loss_wgan_lambda_grow'] = 2.0
FLAGS['loss_wgan_lambda_ignore'] = 1
FLAGS['loss_wgan_use_g_to_one'] = False
FLAGS['loss_wgan_gp_times'] = 1
FLAGS['loss_wgan_gp_use_all'] = False
FLAGS['loss_wgan_gp_bound'] = 5e-2
FLAGS['loss_wgan_gp_mv_decay'] = 0.99

FLAGS['loss_data_term_use_local_weight'] = False
FLAGS['loss_constant_term_use_local_weight'] = False
FLAGS['data_csr_buffer_size'] = 1500
FLAGS['sys_use_all_gpu_memory'] = True
FLAGS['loss_pr'] = (FLAGS['loss_constant_term'] == 'PR' and FLAGS['loss_constant_term_weight'] > 0) or (FLAGS['loss_source_data_term'] == 'PR' and FLAGS['loss_source_data_term_weight'] > 0)
FLAGS['loss_heavy'] = (FLAGS['loss_constant_term_weight'] > 0)

FLAGS['data_augmentation_size'] = 8
FLAGS['data_use_random_pad'] = False
FLAGS['data_train_batch_size'] = 3
FLAGS['load_previous_exp']   = 0
FLAGS['load_previous_epoch'] = 0

FLAGS['process_run_first_testing_epoch'] = True
FLAGS['process_write_test_img_count'] = 498
FLAGS['process_train_log_interval_epoch'] = 20
FLAGS['process_test_log_interval_epoch'] = 2
FLAGS['process_max_epoch'] = 150

FLAGS['format_log_step'] = '%.3f'
FLAGS['format_log_value'] = '{:6.4f}'
if FLAGS['sys_use_unix']:
    FLAGS['path_char'] = '/'
    if FLAGS['sys_is_dgx']:
        FLAGS['path_data'] = '../input/LPGAN'
        FLAGS['path_result_root'] = '../input/LPGAN-Result/%03d-DGX-LPGAN'
    else:
        FLAGS['path_data'] = '/tmp3/nothinglo/dataset/LPGAN'
        FLAGS['path_result_root'] = '/tmp3/nothinglo/dataset/LPGAN-Result/%03d-DGX-LPGAN'
else:
    FLAGS['path_char'] = '\\'
    FLAGS['path_data'] = 'D:\\G\\LPGAN'
    FLAGS['path_result_root'] = 'D:\\LPGAN\\%03d-DGX-LPGAN'

FLAGS['path_result'] = FLAGS['path_result_root'] % FLAGS['num_exp']
FLAGS['load_path'] = FLAGS['path_result_root'] % FLAGS['load_previous_exp'] + FLAGS['path_char']
FLAGS['load_model_path']         = FLAGS['load_path'] + 'model'      + FLAGS['path_char'] + '%s.ckpt' % (FLAGS['format_log_step'] % FLAGS['load_previous_epoch'])
FLAGS['load_train_loss_path']    = FLAGS['load_path'] + 'train_netG_loss' + FLAGS['path_char'] + '%s.txt'  % (FLAGS['format_log_step'] % FLAGS['load_previous_epoch'])
FLAGS['load_train_indices_input_path'] = FLAGS['load_path'] + 'train_ind_input'  + FLAGS['path_char'] + '%s.txt' % (FLAGS['format_log_step'] % FLAGS['load_previous_epoch'])
FLAGS['load_train_indices_label_path'] = FLAGS['load_path'] + 'train_ind_label'  + FLAGS['path_char'] + '%s.txt' % (FLAGS['format_log_step'] % FLAGS['load_previous_epoch'])

FLAGS['load_model_need'] = FLAGS['load_previous_exp'] > 0
FLAGS['process_epoch'] = 0
FLAGS['process_train_drop_summary_step'] = 5
FLAGS['process_test_drop_summary_step'] = 1
FLAGS['process_train_data_loader_count'] = (8 if FLAGS['sys_use_unix'] else 4) if FLAGS['loss_pr'] else 2

# data
FLAGS['data_input_ext'] = '.tif'
FLAGS['data_input_dtype']   = np.uint8
FLAGS['data_label_dtype']   = np.uint8
FLAGS['data_compute_dtype'] = np.float32
FLAGS['data_image_size'] = 512
FLAGS['data_image_channel'] = 3
FLAGS['process_random_seed'] = 2
FLAGS['process_load_test_batch_capacity']  = (8 if FLAGS['sys_use_unix'] else 4) if FLAGS['loss_pr'] else 32
FLAGS['process_load_train_batch_capacity'] = (16 if FLAGS['sys_use_unix'] else 8) if FLAGS['loss_pr'] else 64

# net
FLAGS['net_gradient_clip_value'] = 1e8

# input
FLAGS['folder_input'] = FLAGS['path_data'] + FLAGS['path_char'] + 'input' + FLAGS['path_char']
FLAGS['folder_label'] = FLAGS['path_data'] + FLAGS['path_char'] + 'label' + FLAGS['path_char']
FLAGS['folder_label_HDR'] = FLAGS['path_data'] + FLAGS['path_char'] + 'label_HDR' + FLAGS['path_char']

FLAGS['folder_csrs'] = FLAGS['path_data'] + FLAGS['path_char'] + 'csrs' + FLAGS['path_char']
FLAGS['folder_csrs_rgb'] = FLAGS['path_data'] + FLAGS['path_char'] + 'csrs_rgb' + FLAGS['path_char']
FLAGS['txt_test']   = FLAGS['path_data'] + FLAGS['path_char'] + 'test.txt'
FLAGS['txt_train_input']  = FLAGS['path_data'] + FLAGS['path_char'] + 'train_input.txt'
FLAGS['txt_train_label']  = FLAGS['path_data'] + FLAGS['path_char'] + 'train_label.txt'
if FLAGS['sys_use_unix']:
    FLAGS['folder_test_csrs'] = FLAGS['folder_csrs']
else:
    FLAGS['folder_test_csrs'] = FLAGS['path_data'] + FLAGS['path_char'] + 'test_csrs' + FLAGS['path_char']

# output
FLAGS['folder_model']     = FLAGS['path_result'] + FLAGS['path_char'] + 'model' + FLAGS['path_char']
FLAGS['folder_log']       = FLAGS['path_result'] + FLAGS['path_char'] + 'log' + FLAGS['path_char']
FLAGS['folder_weight']    = FLAGS['path_result'] + FLAGS['path_char'] + 'weight' + FLAGS['path_char']
FLAGS['folder_test_img']  = FLAGS['path_result'] + FLAGS['path_char'] + 'test_img' + FLAGS['path_char']
FLAGS['folder_train_ind_input'] = FLAGS['path_result'] + FLAGS['path_char'] + 'train_ind_input' + FLAGS['path_char']
FLAGS['folder_train_ind_label'] = FLAGS['path_result'] + FLAGS['path_char'] + 'train_ind_label' + FLAGS['path_char']

FLAGS['folder_test_netG_loss']  = FLAGS['path_result'] + FLAGS['path_char'] + 'test_netG_loss' + FLAGS['path_char']
FLAGS['folder_test_netG_psnr1']  = FLAGS['path_result'] + FLAGS['path_char'] + 'test_netG_psnr1' + FLAGS['path_char']
FLAGS['folder_test_netG_psnr2']  = FLAGS['path_result'] + FLAGS['path_char'] + 'test_netG_psnr2' + FLAGS['path_char']
FLAGS['folder_train_netG_loss'] = FLAGS['path_result'] + FLAGS['path_char'] + 'train_netG_loss' + FLAGS['path_char']

FLAGS['netG_mat'] = FLAGS['path_result'] + FLAGS['path_char'] + '%03d-netG.mat' % FLAGS['num_exp']
FLAGS['netD_mat'] = FLAGS['path_result'] + FLAGS['path_char'] + '%03d-netD.mat' % FLAGS['num_exp']
FLAGS['txt_log'] = FLAGS['path_result'] + FLAGS['path_char'] + '%03d-log.txt' % FLAGS['num_exp']
###########################

DIR_LIST = [FLAGS['path_result'], \
FLAGS['folder_model'], FLAGS['folder_log'], FLAGS['folder_weight'], FLAGS['folder_test_img'], \
FLAGS['folder_train_ind_input'], FLAGS['folder_train_ind_label'], \
FLAGS['folder_test_netG_loss'], FLAGS['folder_test_netG_psnr1'], FLAGS['folder_test_netG_psnr2'], FLAGS['folder_train_netG_loss']]

make_dirs(DIR_LIST)
random.seed(FLAGS['process_random_seed'])

test_image_name_list = read_file_to_list(FLAGS['txt_test'],  str, False)
train_image_name_list_input = tmp_train_image_name_list_input = read_file_to_list(FLAGS['txt_train_input'], str, False)
train_image_name_list_label = read_file_to_list(FLAGS['txt_train_label'], str, False)
#train_image_name_list_label_HDR_627 = ["hdr_627_%03d" % (i+1) for i in range(627)]
train_image_name_list_label_HDR_627=[]
#this ooks weird it is steping on train_image_name_list_label
train_image_name_list_label += train_image_name_list_label + train_image_name_list_label_HDR_627

FLAGS['data_test_image_count']  = len(test_image_name_list)
FLAGS['data_train_image_count_input'] = len(train_image_name_list_input)
FLAGS['data_train_image_count_label'] = len(train_image_name_list_label)
FLAGS['data_train_batch_count']  = FLAGS['data_train_image_count_input'] * FLAGS['data_augmentation_size'] // FLAGS['data_train_batch_size']
FLAGS['data_train_sample_count_input'] = FLAGS['data_train_batch_count'] * FLAGS['data_train_batch_size']
FLAGS['data_train_sample_count_label'] = FLAGS['data_train_image_count_label'] * FLAGS['data_augmentation_size']
train_image_name_list_input = np.repeat(train_image_name_list_input, FLAGS['data_augmentation_size'])
train_image_name_list_label = np.repeat(train_image_name_list_label, FLAGS['data_augmentation_size'])
train_image_name_list_input = train_image_name_list_input[:FLAGS['data_train_sample_count_input']]
train_image_name_list_label = train_image_name_list_label[:FLAGS['data_train_sample_count_label']]

class DataFlowMat(object):
    def __init__(self, b):
        self.rect      = [tf.compat.v1.placeholder(tf.int32)                    for _ in range(b)]
        self.rot       = [tf.compat.v1.placeholder(tf.int32)                    for _ in range(b)]
        self.csr_ind   = [tf.compat.v1.placeholder(tf.int64)                    for _ in range(b)]
        self.csr_val   = [tf.compat.v1.placeholder(FLAGS['data_compute_dtype']) for _ in range(b)]
        self.csr_ind_r = [tf.compat.v1.placeholder(tf.int64)                    for _ in range(b)]
        self.csr_val_r = [tf.compat.v1.placeholder(FLAGS['data_compute_dtype']) for _ in range(b)]
        self.csr_ind_g = [tf.compat.v1.placeholder(tf.int64)                    for _ in range(b)]
        self.csr_val_g = [tf.compat.v1.placeholder(FLAGS['data_compute_dtype']) for _ in range(b)]
        self.csr_ind_b = [tf.compat.v1.placeholder(tf.int64)                    for _ in range(b)]
        self.csr_val_b = [tf.compat.v1.placeholder(FLAGS['data_compute_dtype']) for _ in range(b)]
        self.csr_sha   = [tf.compat.v1.placeholder(tf.int64)                    for _ in range(b)]

        self.csr_mat   = [tf.SparseTensor(ind, val, sha) for ind, val, sha in zip(self.csr_ind,   self.csr_val,   self.csr_sha)]
        self.csr_mat_r = [tf.SparseTensor(ind, val, sha) for ind, val, sha in zip(self.csr_ind_r, self.csr_val_r, self.csr_sha)]
        self.csr_mat_g = [tf.SparseTensor(ind, val, sha) for ind, val, sha in zip(self.csr_ind_g, self.csr_val_g, self.csr_sha)]
        self.csr_mat_b = [tf.SparseTensor(ind, val, sha) for ind, val, sha in zip(self.csr_ind_b, self.csr_val_b, self.csr_sha)]
class DataFlow(object):
    def __init__(self, is_training):
        if is_training:
            b = FLAGS['data_train_batch_size']
        else:
            b = 1
        self.input1_src = tf.compat.v1.placeholder(tf.as_dtype(FLAGS['data_input_dtype']), shape=[b, FLAGS['data_image_size'], FLAGS['data_image_size'], FLAGS['data_image_channel']])
        self.input1 = tf.cast(self.input1_src, FLAGS['data_compute_dtype']) / self.input1_src.dtype.max
        self.input2_src = tf.compat.v1.placeholder(tf.as_dtype(FLAGS['data_label_dtype']), shape=[b, FLAGS['data_image_size'], FLAGS['data_image_size'], FLAGS['data_image_channel']])
        self.input2 = tf.cast(self.input2_src, FLAGS['data_compute_dtype']) / self.input2_src.dtype.max

        if is_training:
            self.input1_label_src = tf.compat.v1.placeholder(tf.as_dtype(FLAGS['data_label_dtype']), shape=[b, FLAGS['data_image_size'], FLAGS['data_image_size'], FLAGS['data_image_channel']])
            self.input1_label = tf.cast(self.input1_label_src, FLAGS['data_compute_dtype']) / self.input1_label_src.dtype.max
            self.input2_label_src = tf.compat.v1.placeholder(tf.as_dtype(FLAGS['data_input_dtype']), shape=[b, FLAGS['data_image_size'], FLAGS['data_image_size'], FLAGS['data_image_channel']])
            self.input2_label = tf.cast(self.input2_label_src, FLAGS['data_compute_dtype']) / self.input2_label_src.dtype.max

        self.mat1 = DataFlowMat(b)
        self.mat2 = DataFlowMat(b)

def flatten_list(xs):
    result = []
    if isinstance(xs, (list, tuple)):
        for x in xs:
            result.extend(flatten_list(x))
    else:
        result.append(xs)
    return result
class NetInfo(object):
    def __init__(self, name):
        self.CONV_NETS = []
        seed = FLAGS['process_random_seed']
        ich = FLAGS['data_image_channel']
        if name == "netD":
            init_w = FLAGS['netD_init_weight']
            rw = FLAGS['netD_regularization_weight']
            if FLAGS['netD_init_method'] == "var_scale":
                initializer = tf.compat.v1.keras.initializers.VarianceScaling(init_w, seed=seed)
            elif FLAGS['netD_init_method'] == "rand_uniform":
                initializer = tf.compat.v1.random_uniform_initializer(-init_w*np.sqrt(3), init_w*np.sqrt(3), seed=seed)
            elif FLAGS['netD_init_method'] == "rand_normal":
                initializer = tf.compat.v1.random_normal_initializer(mean=0., stddev=init_w, seed=seed)
            elif FLAGS['netD_init_method'] == "truncated_normal":
                initializer = tf.compat.v1.truncated_normal_initializer(mean=0., stddev=init_w, seed=seed)
            nonlinearity = lrelu_layer(0.2) #prelu_layer()
            norm = in_layer(True, True)
            act = [nonlinearity, norm]
            net_1 = dict(net_name='%s_1' % name, trainable=True)
            net_1['input_index'] = 0
            net_1['layers'] = flatten_list([\
                conv_layer( 3, 1,  16, "SYMMETRIC", initializer), act, \
                conv_layer( 5, 2,  32, "SYMMETRIC", initializer), act, \
                conv_layer( 5, 2,  64, "SYMMETRIC", initializer), act, \
                conv_layer( 5, 2, 128, "SYMMETRIC", initializer), act, \
                conv_layer( 5, 2, 128, "SYMMETRIC", initializer), act, \
                conv_layer( 5, 2, 128, "SYMMETRIC", initializer), act, \
                conv_layer(16, 1,   1,        None, initializer), reduce_mean_layer([1, 2, 3]) \
            ])
            self.CONV_NETS.append(net_1)
        elif name == "netG":
            init_w = FLAGS['netG_init_weight']
            rw = FLAGS['netG_regularization_weight']
            if FLAGS['netG_init_method'] == "var_scale":
                initializer = tf.compat.v1.keras.initializers.VarianceScaling(init_w, seed=seed)
            elif FLAGS['netG_init_method'] == "rand_uniform":
                initializer = tf.compat.v1.random_uniform_initializer(-init_w*np.sqrt(3), init_w*np.sqrt(3), seed=seed)
            elif FLAGS['netG_init_method'] == "rand_normal":
                initializer = tf.compat.v1.random_normal_initializer(mean=0., stddev=init_w, seed=seed)
            elif FLAGS['netG_init_method'] == "truncated_normal":
                initializer = tf.compat.v1.truncated_normal_initializer(mean=0., stddev=init_w, seed=seed)
            nonlinearity = selu_layer() #prelu_layer()
            norm = bn_layer(True, True)
            act = [nonlinearity, norm]
            net_1 = dict(net_name='%s_1' % name, trainable=True)
            net_1['input_index'] = 0
            net_1['layers'] = flatten_list([\
                conv_layer( 3, 1,  16, "SYMMETRIC", initializer), act, \
                conv_layer( 5, 2,  32, "SYMMETRIC", initializer), act, \
                conv_layer( 5, 2,  64, "SYMMETRIC", initializer), act, \
                conv_layer( 5, 2, 128, "SYMMETRIC", initializer), act, \
                conv_layer( 5, 2, 128, "SYMMETRIC", initializer), act, \
            ])
            self.CONV_NETS.append(net_1)

            net_2 = dict(net_name='%s_2' % name, trainable=True)
            net_2['input_index'] = 15
            net_2['layers'] = flatten_list([\
                conv_layer( 5, 2, 128, "SYMMETRIC", initializer), act, \
                conv_layer( 5, 2, 128, "SYMMETRIC", initializer), act, \
                conv_layer( 8, 1, 128,        None, initializer), nonlinearity, \
                conv_layer( 1, 1, 128,        None, initializer) \
            ])
            self.CONV_NETS.append(net_2)
            net_3 = dict(net_name='%s_3' % name, trainable=True)
            net_3['input_index'] = 15
            net_3['layers'] = flatten_list([\
                conv_layer( 3, 1, 128, "SYMMETRIC", initializer), global_concat_layer(24), \
                conv_layer( 1, 1, 128, "SYMMETRIC", initializer), act, \
                conv_layer( 3, 1, 128, "SYMMETRIC", initializer), resize_layer(2, tf.image.ResizeMethod.NEAREST_NEIGHBOR), concat_layer(10), act, \
                conv_layer( 3, 1, 128, "SYMMETRIC", initializer), resize_layer(2, tf.image.ResizeMethod.NEAREST_NEIGHBOR), concat_layer( 7), act, \
                conv_layer( 3, 1,  64, "SYMMETRIC", initializer), resize_layer(2, tf.image.ResizeMethod.NEAREST_NEIGHBOR), concat_layer( 4), act, \
                conv_layer( 3, 1,  32, "SYMMETRIC", initializer), resize_layer(2, tf.image.ResizeMethod.NEAREST_NEIGHBOR), concat_layer( 1), act, \
                conv_layer( 3, 1,  16, "SYMMETRIC", initializer), act, \
                conv_layer( 3, 1, ich, "SYMMETRIC", initializer), res_layer(0, [0, 1, 2]) \
                #, clip_layer() \
            ])
            self.CONV_NETS.append(net_3)
        else:
            assert False, 'net name error'

        self.architecture_log = []
        self.weights = []
        self.parameter_names = []
        self.REGULARIZATION_WEIGHT = rw
        self.base_lr = tf.compat.v1.placeholder(tf.as_dtype(FLAGS['data_compute_dtype']))
        self.OPTIMIZER = tf.compat.v1.train.AdamOptimizer(learning_rate=self.base_lr)
        self.GLOBAL_GRADIENT_CLIPPING = FLAGS['net_gradient_clip_value']
        self.name = name
        self.variable_scope_name = name + '_var_scope'

netD = NetInfo('netD')
netG = NetInfo('netG')
train_df = DataFlow(True)
test_df = DataFlow(False)

def prepare_csr_dict(csr_dict, read_data):
    keys = []
    if FLAGS['loss_pr']:
        #this is tempoal coment
        #test_count = FLAGS['data_test_image_count'] if FLAGS['process_write_test_img_count'] == 0 or FLAGS['process_train_data_loader_count'] <= 0 else FLAGS['process_write_test_img_count']
        test_count = FLAGS['data_test_image_count']
        total_name_list = tmp_train_image_name_list_input + test_image_name_list[:test_count]
        d_i = 3 if FLAGS['loss_photorealism_is_our'] else 1
        end_i = min(len(total_name_list), FLAGS['data_csr_buffer_size'] // d_i)
        for i in range(end_i):
            file_name = total_name_list[i]
            if FLAGS['loss_photorealism_is_our']:
                if read_data:
                    input_csr_r = io.loadmat(FLAGS['folder_csrs_rgb'] + file_name + '_r.mat')['CSR']
                    input_csr_g = io.loadmat(FLAGS['folder_csrs_rgb'] + file_name + '_g.mat')['CSR']
                    input_csr_b = io.loadmat(FLAGS['folder_csrs_rgb'] + file_name + '_b.mat')['CSR']
                    csr_dict.update({file_name + '_r':input_csr_r, file_name + '_g':input_csr_g, file_name + '_b':input_csr_b})
                keys.extend([file_name + '_r', file_name + '_g', file_name + '_b'])
            else:
                if read_data:
                    input_csr = io.loadmat(FLAGS['folder_csrs'] + file_name + '.mat')['CSR']
                    csr_dict.update({file_name:input_csr})
                keys.extend([file_name])
    return keys
csr_dict_keys = prepare_csr_dict(None, False)
csr_dict = {}
##########################################################################
class DataMat(object):
    def __init__(self, pix_c):
        self.csr_names, self.csr_ind_r, self.csr_val_r, self.csr_ind_g, self.csr_val_g, self.csr_ind_b, self.csr_val_b, self.csr_ind, self.csr_val = [], [], [], [], [], [], [], [], []
        self.pix_c = pix_c
    def load(self, file_name, csr_dict_keys):
        if FLAGS['loss_photorealism_is_our']:
            if file_name + '_r' in csr_dict_keys:
                self.csr_ind_r = None
                self.csr_val_r = None
                self.csr_ind_g = None
                self.csr_val_g = None
                self.csr_ind_b = None
                self.csr_val_b = None
                self.csr_names.extend([file_name + '_r', file_name + '_g', file_name + '_b'])
            else:
                input_csr_r = io.loadmat(FLAGS['folder_csrs_rgb'] + file_name + '_r.mat')['CSR']
                input_csr_g = io.loadmat(FLAGS['folder_csrs_rgb'] + file_name + '_g.mat')['CSR']
                input_csr_b = io.loadmat(FLAGS['folder_csrs_rgb'] + file_name + '_b.mat')['CSR']
                self.csr_ind_r = input_csr_r[:, :2] - 1
                self.csr_val_r = input_csr_r[:, -1]
                self.csr_ind_g = input_csr_g[:, :2] - 1
                self.csr_val_g = input_csr_g[:, -1]
                self.csr_ind_b = input_csr_b[:, :2] - 1
                self.csr_val_b = input_csr_b[:, -1]
            self.pix_c = self.pix_c + 1
        else:
            if file_name in csr_dict_keys:
                self.csr_ind = None
                self.csr_val = None
                self.csr_names.append(file_name)
            else:
                input_csr = io.loadmat(FLAGS['folder_csrs'] + file_name + '.mat')['CSR']
                self.csr_ind = input_csr[:, :2] - 1
                self.csr_val = input_csr[:, -1]
    def update(self, data_batch, i):
        data_batch['csr_ind%d'   % i].append(self.csr_ind)
        data_batch['csr_val%d'   % i].append(self.csr_val)
        data_batch['csr_sha%d'   % i].append([self.pix_c, self.pix_c])
        data_batch['csr_ind_r%d' % i].append(self.csr_ind_r)
        data_batch['csr_val_r%d' % i].append(self.csr_val_r)
        data_batch['csr_ind_g%d' % i].append(self.csr_ind_g)
        data_batch['csr_val_g%d' % i].append(self.csr_val_g)
        data_batch['csr_ind_b%d' % i].append(self.csr_ind_b)
        data_batch['csr_val_b%d' % i].append(self.csr_val_b)
        data_batch['csr_names%d' % i].append(self.csr_names)

class DataLoader(object):
    def __init__(self):
        self.test_input_batch_queue = Queue(FLAGS['process_load_test_batch_capacity'])
        self.test_label_queue       = Queue(FLAGS['process_load_test_batch_capacity'])

        self.test_data_load_process = Process(target=self.enqueue_test_batch, args=(self.test_input_batch_queue, self.test_label_queue))
        self.test_data_load_process.daemon = True
        self.test_data_load_process.start()

        self.train_indices_queue = Queue(FLAGS['data_train_sample_count_input'])

        self.train_batch_queue = Queue(FLAGS['process_load_train_batch_capacity'])
        self.train_data_load_process_list = []
        for _ in range(FLAGS['process_train_data_loader_count']):
            data_load_process = Process(target=self.enqueue_train_batch, args=(self.train_batch_queue, self.train_indices_queue))
            data_load_process.daemon = True
            data_load_process.start()
            self.train_data_load_process_list.append(data_load_process)
    def enqueue_test_batch(self, input_queue, label_queue):
        while True:
            #temporaly removed
            #test_count = FLAGS['data_test_image_count'] if FLAGS['process_write_test_img_count'] == 0 or FLAGS['process_train_data_loader_count'] <= 0 else FLAGS['process_write_test_img_count']
            test_count = FLAGS['data_test_image_count']
            for i in range(test_count):
                file_name = test_image_name_list[i]
                input_img = cv2.imread(FLAGS['folder_input'] + file_name + FLAGS['data_input_ext'], -1)
                label_img = cv2.imread(FLAGS['folder_label'] + file_name + FLAGS['data_input_ext'], -1)
                
                
                print(file_name)

                print(input_img.shape)
                print(label_img.shape)



                pix_c = input_img.shape[0] * input_img.shape[1]
                rot = 0
                input_img, _, rect = random_pad_to_size(input_img, FLAGS['data_image_size'], None, True, False)
                label_img, _, _    = random_pad_to_size(label_img, FLAGS['data_image_size'], None, True, False)

                data_batch = dict(rect1=[rect], rect2=[rect], rot1=[rot], rot2=[rot], \
                    csr_ind1=[], csr_val1=[], csr_sha1=[], csr_ind_r1=[], csr_val_r1=[], csr_ind_g1=[], csr_val_g1=[], csr_ind_b1=[], csr_val_b1=[], csr_names1=[], \
                    csr_ind2=[], csr_val2=[], csr_sha2=[], csr_ind_r2=[], csr_val_r2=[], csr_ind_g2=[], csr_val_g2=[], csr_ind_b2=[], csr_val_b2=[], csr_names2=[])

                mat1 = DataMat(pix_c)
                # mat2 = DataMat(pix_c)

                if FLAGS['loss_source_data_term'] == 'PR' and FLAGS['loss_source_data_term_weight'] > 0:
                    mat1.load(file_name, csr_dict_keys)
                    mat1.update(data_batch, 1)
                # if FLAGS['loss_target_data_term'] == 'PR' and FLAGS['loss_target_data_term_weight'] > 0:
                #     mat2.load(file_name, csr_dict_keys)
                #     mat2.update(data_batch, 2)

                input_queue.put([[input_img], data_batch])
                label_queue.put([label_img])

    def enqueue_train_indices_by_data(self, indices_input, indices_label, start_iter):
        if len(indices_input) != len(indices_label):
            indices_input, indices_label = self.enqueue_train_indices_by_shuffle_method(indices_input, indices_label, start_iter)
        assert(len(indices_input) == len(indices_label))
        for b in range(start_iter * FLAGS['data_train_batch_size'], len(indices_input), FLAGS['data_train_batch_size']):
            self.train_indices_queue.put((indices_input[b:b+FLAGS['data_train_batch_size']], indices_label[b:b+FLAGS['data_train_batch_size']]))
        return indices_input, indices_label

    def enqueue_train_indices_by_shuffle_method(self, indices_input, indices_label, start_iter):
        random.shuffle(indices_input)
        if FLAGS['data_train_sample_count_input'] < FLAGS['data_train_sample_count_label']:
            indices_label = list(range(FLAGS['data_train_sample_count_label']))
            random.shuffle(indices_label)
            indices_label = indices_label[:len(indices_input)]
        elif FLAGS['data_train_sample_count_input'] > FLAGS['data_train_sample_count_label']:
            copy_time = FLAGS['data_train_sample_count_input'] // FLAGS['data_train_sample_count_label']
            if FLAGS['data_train_sample_count_input'] % FLAGS['data_train_sample_count_label'] != 0:
                copy_time = copy_time + 1
            indices_label = list(range(FLAGS['data_train_sample_count_label']))
            indices_label = np.repeat(indices_label, copy_time)
            random.shuffle(indices_label)
            indices_label = indices_label[:len(indices_input)]
        else:
            random.shuffle(indices_label)
        return self.enqueue_train_indices_by_data(indices_input, indices_label, start_iter)

    def enqueue_train_batch(self, batch_queue, indices_queue):
        def get_patch(file_name_input, file_name_label):
            input_img = cv2.imread(FLAGS['folder_input'] + file_name_input + FLAGS['data_input_ext'], -1)
            if os.path.exists(FLAGS['folder_label'] + file_name_label + FLAGS['data_input_ext']):
                label_img = cv2.imread(FLAGS['folder_label'] + file_name_label + FLAGS['data_input_ext'], -1)
            else:
                label_img = cv2.imread(FLAGS['folder_label_HDR'] + file_name_label + FLAGS['data_input_ext'], -1)

            input_label_img = cv2.imread(FLAGS['folder_label'] + file_name_input + FLAGS['data_input_ext'], -1)
            #label_input_img = cv2.imread(FLAGS['folder_input'] + file_name_label + FLAGS['data_input_ext'], -1)
            print(FLAGS['folder_input'] + file_name_input + FLAGS['data_input_ext'])
            print(input_img.shape)
            print(FLAGS['folder_label'] + file_name_label + FLAGS['data_input_ext'])
            print(label_img.shape)
            pix_c1 = input_img.shape[0] * input_img.shape[1]
            pix_c2 = label_img.shape[0] * label_img.shape[1]

            mat1 = DataMat(pix_c1)
            # mat2 = DataMat(pix_c2)

            if FLAGS['loss_source_data_term'] == 'PR' and FLAGS['loss_source_data_term_weight'] > 0:
                mat1.load(file_name_input, csr_dict_keys)
                mat1.update(data_batch, 1)
            # if FLAGS['loss_target_data_term'] == 'PR' and FLAGS['loss_target_data_term_weight'] > 0:
            #     mat2.load(file_name_label, csr_dict_keys)
            #     mat2.update(data_batch, 2)

            return input_img, label_img, input_label_img#, label_input_img
        while True:
            indices_1, indices_2 = indices_queue.get()
            data_batch = dict(input=[], label=[], input_label=[], label_input=[], rect1=[], rect2=[], rot1=[], rot2=[], \
                    csr_ind1=[], csr_val1=[], csr_sha1=[], csr_ind_r1=[], csr_val_r1=[], csr_ind_g1=[], csr_val_g1=[], csr_ind_b1=[], csr_val_b1=[], csr_names1=[], \
                    csr_ind2=[], csr_val2=[], csr_sha2=[], csr_ind_r2=[], csr_val_r2=[], csr_ind_g2=[], csr_val_g2=[], csr_ind_b2=[], csr_val_b2=[], csr_names2=[])

            for i, j in zip(indices_1, indices_2):
                #input_img, label_img, input_label_img, label_input_img = get_patch(train_image_name_list_input[i], train_image_name_list_label[j])
                input_img, label_img, input_label_img = get_patch(train_image_name_list_input[i], train_image_name_list_label[j])
                rot1 = i % FLAGS['data_augmentation_size']
                rot2 = j % FLAGS['data_augmentation_size']
                
                input_img, _, rect1 = data_augmentation(input_img, rot1, FLAGS['data_augmentation_size'], FLAGS['data_image_size'], True, FLAGS['data_use_random_pad'])
                label_img, _, rect2 = data_augmentation(label_img, rot2, FLAGS['data_augmentation_size'], FLAGS['data_image_size'], True, FLAGS['data_use_random_pad'])
                input_label_img, _, _ = data_augmentation(input_label_img, rot1, FLAGS['data_augmentation_size'], FLAGS['data_image_size'], True, FLAGS['data_use_random_pad'])
                #label_input_img, _, _ = data_augmentation(label_input_img, rot2, FLAGS['data_augmentation_size'], FLAGS['data_image_size'], True, FLAGS['data_use_random_pad'])
                assert FLAGS['data_use_random_pad'] == False, 'FLAGS[data_use_random_pad] should be False'
                data_batch['input'].append(input_img)
                data_batch['label'].append(label_img)
                data_batch['input_label'].append(input_label_img)
                #data_batch['label_input'].append(label_input_img)
                data_batch['rect1'].append(rect1)
                data_batch['rot1'].append(rot1)
                data_batch['rect2'].append(rect2)
                data_batch['rot2'].append(rot2)
            batch_queue.put(data_batch)
    def get_next_test_input_batch(self):
        return self.test_input_batch_queue.get()
    def get_next_test_label(self):
        return self.test_label_queue.get()
    def get_next_train_batch(self):
        return self.train_batch_queue.get()

