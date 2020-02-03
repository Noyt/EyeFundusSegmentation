import numpy as np
import os
import cv2
from utils.fundus_process import ROI
from tqdm import tqdm


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def get_most_recent_file(dirpath):
    files = [os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(dirpath)) for f in fn]
    files.sort(key=lambda x: os.path.getmtime(x))
    return files[-1]


def save_numpy(arr, path):
    create_folder(path)
    np.save(path, arr)


class StandardDB:
    def __init__(self, input_path,
                 output_path,
                 image_folder='images/',
                 red_folders=None,
                 bright_folders=None,
                 create_empty_instances=False,
                 verbose=True,
                 crop_threshold=5,
                 preprocess_name=None,
                 save_as='.png'):

        self.input_folder = input_path
        self.output_folder = output_path

        if red_folders is None:
            red_folders = []
        if bright_folders is None:
            bright_folders = []

        if not isinstance(red_folders, list):
            red_folders = [red_folders]

        if not isinstance(bright_folders, list):
            bright_folders = [bright_folders]

        self.preprocess_name = preprocess_name
        self.create_empty_instances = create_empty_instances
        self.verbose = verbose
        self.crop_threshold = crop_threshold
        self.save_as = save_as

        self.red_folders = [os.path.join(input_path, _) for _ in red_folders]
        self.bright_folders = [os.path.join(input_path, _) for _ in bright_folders]
        self.image_folder = [os.path.join(input_path, image_folder)]

        self.list_images = self.find_common_images(self.image_folder + self.red_folders + self.bright_folders)

        if verbose:
            print('Found %i commons images in each folders' % len(self.list_images))

    def filenames_wo_ext(self, files_list):
        return [os.path.splitext(os.path.basename(_))[0] for _ in files_list if _ != '.directory']

    def find_common_images(self, folders):
        common_images = set(self.filenames_wo_ext(os.listdir(folders[0])))
        for f in folders[1:]:
            list_files = self.filenames_wo_ext(os.listdir(f))
            if self.preprocess_name is None:
                common_images = common_images & set(list_files)
            else:
                common_images = common_images & set([self.preprocess_name(_) for _ in list_files])
        return list(common_images)

    def create_folders(self):
        create_folder(os.path.join(self.output_folder, 'images/'))
        create_folder(os.path.join(self.output_folder, 'red/'))
        create_folder(os.path.join(self.output_folder, 'bright/'))

    def find_original_name(self, short_name, folder):
        list_files_in_folder = os.listdir(folder)
        candidates = [_ for _ in list_files_in_folder if _.startswith(short_name)]
        if len(candidates) != 1:
            raise FileNotFoundError("Didn't find a match for shortname %s in folder %s" % (short_name, folder))
        else:
            return candidates[0]

    def process(self):
        self.create_folders()
        if self.verbose:
            pbar = tqdm(total=len(self.list_images))

        for img_name in self.list_images:
            img_path = os.path.join(self.image_folder[0], self.find_original_name(img_name, self.image_folder[0]))
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            roi_mask = ROI(img, self.crop_threshold)

            not_null_pixels = np.nonzero(roi_mask)
            x_range = (np.min(not_null_pixels[1]), np.max(not_null_pixels[1]))
            y_range = (np.min(not_null_pixels[0]), np.max(not_null_pixels[0]))
            img = img[y_range[0]:y_range[1], x_range[0]:x_range[1]]

            h, w, c = img.shape
            output_shape = max(h, w)
            padding_h = output_shape - h
            padding_w = output_shape - w
            padding = [(int(np.floor(padding_h / 2)), int(np.ceil(padding_h / 2))),
                       (int(np.floor(padding_w / 2)), int(np.ceil(padding_w / 2)))]

            img = np.pad(img, padding + [(0, 0)], 'constant')

            cv2.imwrite(os.path.join(os.path.join(self.output_folder, 'images/'),
                                     img_name + self.save_as), img)

            def crop_and_join(folders):
                lesion_mask = np.zeros((output_shape, output_shape), dtype=np.bool)
                for f in folders:
                    img_path = os.path.join(f, self.find_original_name(img_name, f))
                    lesion = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)[y_range[0]:y_range[1], x_range[0]:x_range[1]]
                    lesion = np.pad(lesion, padding, 'constant')
                    lesion_mask = np.logical_or(lesion_mask, lesion)

                return 255 * lesion_mask.astype(np.uint8)

            if self.red_folders or self.create_empty_instances:
                red_lesions_img = crop_and_join(self.red_folders)
                cv2.imwrite(os.path.join(os.path.join(self.output_folder, 'red/'),
                                         img_name + self.save_as),
                            red_lesions_img)

            if self.bright_folders or self.create_empty_instances:
                bright_lesions_img = crop_and_join(self.bright_folders)
                cv2.imwrite(os.path.join(os.path.join(self.output_folder, 'bright/'), img_name + self.save_as),
                            bright_lesions_img)
            if self.verbose:
                pbar.update(1)
