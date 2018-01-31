import os


class DataProcess(object):
    def __init__(self, data_dir):
        self.categories_filenames = os.listdir(data_dir)
        self.data_dir = data_dir
        self.categories_names_cates = []

    def decode(self):

        for category_filename in self.categories_filenames:
            category = category_filename[:-4]
            category_filepath = "%s/%s" % (self.data_dir, category_filename)

            with open(category_filepath, "r", encoding="utf8") as f:
                category_lines = f.readlines()
            category_names = [category_line[:-1] for category_line in category_lines]
            category_names_cates = list(zip(category_names, [category] * len(category_names)))

            self.categories_names_cates += category_names_cates
            print("DataProcess.process: category: %s, length: %s" % (category, len(category_names_cates)))
        return self.categories_names_cates

    def save(self, save_path, save_type="csv"):
        if save_type == "csv":
            save_data_tmp = ["#".join(category_name_cate) + "\n" for category_name_cate in self.categories_names_cates]
            with open(save_path, "w", encoding="utf8") as f:
                f.writelines(save_data_tmp)
        else:
            raise NotImplementedError("other save_type is not supported yet.")


if __name__ == "__main__":
    data_process = DataProcess("./names")
    data_process.decode()
    data_process.save("./names.txt")
