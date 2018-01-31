class BaseFunction(object):

    @staticmethod
    def iter_sum(lst):
        tmp = lst[0]
        for l in lst[1:]:
            tmp += l
        return tmp
