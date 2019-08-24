from ops import *
from utils import *
from glob import glob
import time
from tensorflow.contrib.data import prefetch_to_device, shuffle_and_repeat, map_and_batch
import numpy as np
import argparse

import cv2

cascade_path = "./haarcascade_frontalface_alt.xml"


class UGATIT(object) :
    def __init__(self, sess, args):
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.sess = sess
        self.checkpoint_dir = args.checkpoint_dir
        self.dataset_name = args.dataset

        self.gan_type = args.gan_type

        self.batch_size = args.batch_size

        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight
        self.smoothing = args.smoothing

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis
        self.n_critic = args.n_critic
        self.sn = args.sn

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Generator
    ##################################################################################

    def generator(self, x_init, reuse=False, scope="generator"):
        channel = self.ch
        with tf.variable_scope(scope, reuse=reuse) :
            x = conv(x_init, channel, kernel=7, stride=1, pad=3, pad_type='reflect', scope='conv')
            x = instance_norm(x, scope='ins_norm')
            x = relu(x)

            # Down-Sampling
            for i in range(2) :
                x = conv(x, channel*2, kernel=3, stride=2, pad=1, pad_type='reflect', scope='conv_'+str(i))
                x = instance_norm(x, scope='ins_norm_'+str(i))
                x = relu(x)

                channel = channel * 2

            # Down-Sampling Bottleneck
            for i in range(self.n_res):
                x = resblock(x, channel, scope='resblock_' + str(i))


            # Class Activation Map
            cam_x = global_avg_pooling(x)
            cam_gap_logit, cam_x_weight = fully_connected_with_w(cam_x, scope='CAM_logit')
            x_gap = tf.multiply(x, cam_x_weight)

            cam_x = global_max_pooling(x)
            cam_gmp_logit, cam_x_weight = fully_connected_with_w(cam_x, reuse=True, scope='CAM_logit')
            x_gmp = tf.multiply(x, cam_x_weight)


            cam_logit = tf.concat([cam_gap_logit, cam_gmp_logit], axis=-1)
            x = tf.concat([x_gap, x_gmp], axis=-1)

            x = conv(x, channel, kernel=1, stride=1, scope='conv_1x1')
            x = relu(x)

            heatmap = tf.squeeze(tf.reduce_sum(x, axis=-1))

            # Gamma, Beta block
            gamma, beta = self.MLP(x, reuse=reuse)

            # Up-Sampling Bottleneck
            for i in range(self.n_res):
                x = adaptive_ins_layer_resblock(x, channel, gamma, beta, smoothing=self.smoothing, scope='adaptive_resblock' + str(i))

            # Up-Sampling
            for i in range(2) :
                x = up_sample(x, scale_factor=2)
                x = conv(x, channel//2, kernel=3, stride=1, pad=1, pad_type='reflect', scope='up_conv_'+str(i))
                x = layer_instance_norm(x, scope='layer_ins_norm_'+str(i))
                x = relu(x)

                channel = channel // 2


            x = conv(x, channels=3, kernel=7, stride=1, pad=3, pad_type='reflect', scope='G_logit')
            x = tanh(x)

            return x, cam_logit, heatmap

    def MLP(self, x, use_bias=True, reuse=False, scope='MLP'):
        channel = self.ch * self.n_res

        if self.light :
            x = global_avg_pooling(x)

        with tf.variable_scope(scope, reuse=reuse):
            for i in range(2) :
                x = fully_connected(x, channel, use_bias, scope='linear_' + str(i))
                x = relu(x)


            gamma = fully_connected(x, channel, use_bias, scope='gamma')
            beta = fully_connected(x, channel, use_bias, scope='beta')

            gamma = tf.reshape(gamma, shape=[self.batch_size, 1, 1, channel])
            beta = tf.reshape(beta, shape=[self.batch_size, 1, 1, channel])

            return gamma, beta

    ##################################################################################
    # Model
    ##################################################################################

    def generate_a2b(self, x_A, reuse=False):
        out, cam, _ = self.generator(x_A, reuse=reuse, scope="generator_B")

        return out, cam

    # def generate_b2a(self, x_B, reuse=False):
    #     out, cam, _ = self.generator(x_B, reuse=reuse, scope="generator_A")

    #     return out, cam

    def build_model(self):
        """ Test """
        self.test_domain_A = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_domain_A')
        # self.test_domain_B = tf.placeholder(tf.float32, [1, self.img_size, self.img_size, self.img_ch], name='test_domain_B')

        self.test_fake_B, _ = self.generate_a2b(self.test_domain_A)
        # self.test_fake_A, _ = self.generate_b2a(self.test_domain_B)

    @property
    def model_dir(self):
        n_res = str(self.n_res) + 'resblock'
        n_dis = str(self.n_dis) + 'dis'

        if self.smoothing :
            smoothing = '_smoothing'
        else :
            smoothing = ''

        if self.sn :
            sn = '_sn'
        else :
            sn = ''

        return "{}_{}_{}_{}_{}_{}_{}_{}_{}_{}{}{}".format(self.model_name, self.dataset_name,
                                                         self.gan_type, n_res, n_dis,
                                                         self.n_critic,
                                                         self.adv_weight, self.cycle_weight, self.identity_weight, self.cam_weight, sn, smoothing)


    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            counter = int(ckpt_name.split('-')[-1])
            print(" [*] Success to read {}".format(ckpt_name))
            return True, counter
        else:
            print(" [*] Failed to find a checkpoint")
            return False, 0

    def movie(self):
        if args.device == 'normal_cam':
            cam = cv2.VideoCapture(0)
            # cam = cv2.VideoCapture("test.mp4")
        elif args.device == 'jetson_nano_raspi_cam':
            GST_STR = 'nvarguscamerasrc \
                ! video/x-raw(memory:NVMM), width=3280, height=2464, format=(string)NV12, framerate=(fraction)30/1 \
                ! nvvidconv ! video/x-raw, width=(int)1920, height=(int)1080, format=(string)BGRx \
                ! videoconvert \
                ! appsink'
            cam = cv2.VideoCapture(GST_STR, cv2.CAP_GSTREAMER) # Raspi cam
        else:
            print('wrong device')
            sys.exit()

        tf.global_variables_initializer().run()

        self.saver = tf.train.Saver()
        could_load, checkpoint_counter = self.load(self.checkpoint_dir)

        if could_load :
            print(" [*] Load SUCCESS")
        else :
            print(" [!] Load failed...")

        count_max = 0
        count = 0
        window_name = 'selfie2anime'
        if args.fullscreen:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        while True:
            ret, img = cam.read()
            if not ret:
                print('error')
                break
            key = cv2.waitKey(1)
            if key == 27: # when ESC key is pressed break
                break

            count += 1
            if count > count_max:
                image_np = img
                # image_np = img[:,:,::-1] # convert bgr to rgb

                # detect face and crop face region of image
                image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                cascade = cv2.CascadeClassifier(cascade_path)
                facerect = cascade.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=1, minSize=(1, 1))

                if len(facerect) > 0:
                    face_size_max = 0
                    for rect in facerect:
                        face_size = rect[2] + rect[3]
                        if face_size > face_size_max:
                            face_size_max = face_size
                            face_x_margin = rect[2] * 0.3
                            face_y_margin_top = rect[3] * 0.5
                            face_y_margin_bottom = rect[3] * 0.1
                            face_x_min = max(0, (int)(rect[0]-face_x_margin))
                            face_x_max = min(image_np.shape[1], (int)((rect[0]+rect[2])+face_x_margin))
                            face_y_min = max(0, (int)(rect[1]-face_y_margin_top))
                            face_y_max = min(image_np.shape[0], (int)((rect[1]+rect[3])+face_y_margin_bottom))

                    image_np = image_np[face_y_min:face_y_max, face_x_min:face_x_max]

                    # selfie2anime
                    test_file_name = 'test.jpg'
                    cv2.imwrite(test_file_name, image_np)
                    sample_file = test_file_name
                    sample_image = np.asarray(load_test_data(sample_file, size=self.img_size))
                    fake_img = self.sess.run(self.test_fake_B, feed_dict = {self.test_domain_A : sample_image})
                    fake_img = inverse_transform(fake_img)
                    fake_img = merge(fake_img, [1,1])
                    fake_img = cv2.cvtColor(fake_img.astype('uint8'), cv2.COLOR_RGB2BGR)
                    cv2.imshow(window_name, fake_img)

        print("end")
        cam.release()
        cv2.destroyAllWindows()

def parse_args():
    desc = "Tensorflow implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--light', type=str2bool, default=False, help='[U-GAT-IT full version / U-GAT-IT light version]')
    parser.add_argument('--dataset', type=str, default='selfie2anime', help='dataset_name')

    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')

    parser.add_argument('--adv_weight', type=int, default=1, help='Weight about GAN')
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight about Cycle')
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight about Identity')
    parser.add_argument('--cam_weight', type=int, default=1000, help='Weight about CAM')
    parser.add_argument('--gan_type', type=str, default='lsgan', help='[gan / lsgan / wgan-gp / wgan-lp / dragan / hinge]')

    parser.add_argument('--smoothing', type=str2bool, default=True, help='AdaLIN smoothing effect')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')
    parser.add_argument('--n_critic', type=int, default=1, help='The number of critic')
    parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')

    parser.add_argument('--checkpoint_dir', type=str, default='checkpoint',
                        help='Directory name to save the checkpoints')

    parser.add_argument('-d', '--device', type=str, default='normal_cam',
                        help='Device Name normal_cam / jetson_nano_raspi_cam')
    parser.add_argument('--fullscreen', type=bool, default=False,
                        help='Full screen mode')



    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()
    if args is None:
      exit()

    # open session
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        gan = UGATIT(sess, args)

        # build graph
        gan.build_model()

        # show network architecture
        show_all_variables()

        gan.movie()