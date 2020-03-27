import os, sys
import tensorflow as tf


from MODEL import *
from FUNCTION import *
from PREPROCESSING import *
from DATA_infer  import *
import argparse



def parse_args():
    
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=64)
    
    parser.add_argument('--num_words', type=int)
    parser.add_argument('--word_index_len', type=int)
    parser.add_argument('--labels_index_len', type=int)
    parser.add_argument('--embedding_dim', type=int)
    parser.add_argument('--max_sequence_len', type=int)
    
    # data directories
    parser.add_argument('--inference_dir', type=str, default='../input/LPGAN/input/')
    parser.add_argument('--output_dir', type=str, default='../input/LPGAN-result/')
    
    # embedding directory
    parser.add_argument('--embedding', type=str, default=os.environ.get('SM_CHANNEL_EMBEDDING'))
    
    # model directory: we will use the default set by SageMaker, /opt/ml/model
    #parser.add_argument('--inference_dir', type=str, default=)
    
    return parser.parse_known_args()



tf.compat.v1.disable_eager_execution()
print(tf.config.list_physical_devices('GPU'))

print(current_time() + ', exp = %s, load_model path = %s' % (FLAGS['num_exp'], os.path.dirname(os.path.abspath(__file__))))
os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS['num_gpu']
netG_act_o = dict(size=1, index=0)
etG_act_o = dict(size=1, index=0)

netG = NetInfo('netG-%d' % FLAGS['num_exp'])
test_df = DataFlow()
with tf.compat.v1.name_scope(netG.name):
    with tf.compat.v1.variable_scope(netG.variable_scope_name) as scope_full:
        with tf.compat.v1.variable_scope(netG.variable_scope_name + 'A') as scopeA:
            netG_test_output1 = model(netG, test_df.input1, False, netG_act_o, is_first=True)

assert len(netG.weights) == len(netG.parameter_names), 'len(weights) != len(parameters)'
saver = tf.compat.v1.train.Saver(var_list=netG.weights, max_to_keep=None)

with tf.compat.v1.name_scope("Loss"):
    netG_test_output1_crop = tf_crop_rect(netG_test_output1, test_df.mat1, 0)

with tf.compat.v1.name_scope("Resize"):
    tf_input_img_ori = tf.compat.v1.placeholder(tf.uint8, shape=[None, None, 3])
    tf_img_new_h = tf.compat.v1.placeholder(tf.int32)
    tf_img_new_w = tf.compat.v1.placeholder(tf.int32)
    tf_resize_img = tf.image.resize(images=tf_input_img_ori, size=[tf_img_new_h, tf_img_new_w], method=tf.image.ResizeMethod.AREA)

sess_config = tf.compat.v1.ConfigProto(log_device_placement=False)
sess_config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session(config=sess_config)
sess.run(tf.compat.v1.global_variables_initializer())
sess.run(tf.compat.v1.local_variables_initializer())
saver.restore(sess, FLAGS['load_model_path_new'])

#saver.restore(sess,'/home/felipe/gans_enhancer/Online_Demo_Models_Deep-Photo-Enhancer/LPGAN_exp_G3_999/model')

def checkValidImg(input_img):
    print(current_time() + ', [checkValidImg]')
    if input_img is None:
        print(current_time() + ', img is None')
        return None
    if len(input_img.shape) != 3:
        print(current_time() + ', len(shape) != 3')
        return None
    if input_img.shape[2] != 3:
        print(current_time() + ', shape[2] != 3')
        return None
    if input_img.dtype != np.uint8:
        print(current_time() + ', img.dtype != uint8')
        return None
    return True

def normalizeImage(img, max_length):
    print(current_time() + ', [normalizeImage]')
    [height, width, channels] = img.shape
    print(current_time() + ', original shape = [%d, %d, %d]' % (height, width, channels))
    max_l = max(height, width)

    is_need_resize = max_l != FLAGS['data_image_size']
    if is_need_resize:
        use_gpu = False 
        if use_gpu and is_downsample:
            # gpu
            new_h, new_w, is_normalize = get_normalize_size_shape_method(img, max_length)
            if not is_normalize:
                dict_d = [img, new_h, new_w]
                dict_t = [tf_input_img_ori, tf_img_new_h, tf_img_new_w]
                img = sess.run(tf_resize_img, feed_dict={t:d for t, d in zip(dict_t, dict_d)})
        else:
            # cpu
            img = cpu_normalize_image(img, max_length)
    return img

def getInputPhoto(file_name):
    print(current_time() + ', [getInputPhoto]: file_name = %s' % (FLAGS['folder_input'] + file_name))
    file_name_without_ext = os.path.splitext(file_name)[0]
    input_img = cv2.imread(FLAGS['folder_inference'] + file_name, 1)
    os.remove(FLAGS['folder_inference'] + file_name)
    if checkValidImg(input_img):
        h, w, _ = input_img.shape
        resize_input_img = normalizeImage(input_img, FLAGS['data_max_image_size']) if max(h, w) > FLAGS['data_max_image_size'] else input_img
        file_name = file_name_without_ext + FLAGS['data_output_ext']
        #cv2.imwrite(FLAGS['folder_input'] + file_name, resize_input_img)
        cv2.imwrite(FLAGS['folder_infeence'] + file_name_without_ext + '.jpg', resize_input_img)
        os.rename(FLAGS['folder_inference'] + file_name_without_ext + '.jpg', FLAGS['folder_inference'] + file_name)
        return file_name
    else:
        return None

def processImg(file_in_name, file_out_name_without_ext):
    print(current_time() + ', [processImg]: file_name = %s' % (FLAGS['folder_inference'] + file_in_name))
    input_img = cv2.imread(FLAGS['folder_inference'] + file_in_name, -1)
    resize_input_img = normalizeImage(input_img, FLAGS['data_image_size'])
    resize_input_img, _, _ = random_pad_to_size(resize_input_img, FLAGS['data_image_size'], None, True, False)
    input_img, _, rect = random_pad_to_size(input_img, FLAGS['data_image_size'], None, True, False)
    input_img = input_img[None, :, :, :]
    dict_d = [input_img, rect, 0]
    dict_t = [test_df.input1_src] + \
        test_df.mat1.rect + test_df.mat1.rot
    enhance_test_img = sess.run(netG_test_output1_crop, feed_dict={t:d for t, d in zip(dict_t, dict_d)})
    enhance_test_img = safe_casting(enhance_test_img * tf.as_dtype(FLAGS['data_input_dtype']).max, FLAGS['data_input_dtype'])
    enhanced_img_file_name = file_out_name_without_ext + FLAGS['data_output_ext']
    enhance_img_file_path = FLAGS['folder_test_img'] + enhanced_img_file_name
    #try:
    #    print(current_time() + ', try remove file path = %s' % enhance_img_file_path)
    #    os.remove(enhance_img_file_path)
    #except OSError as e:
    #    print(current_time() + ', remove fail, error = %s' % e.strerror)
    cv2.imwrite(enhance_img_file_path, enhance_test_img)
    return enhanced_img_file_name

if __name__=='__main__':
    args, _ = parse_args()

    FLAGS['inference_folder'] = args.inference_dir
    print('processing')
    processImg('a0002.tif','totaltest')
