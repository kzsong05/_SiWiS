from ping3 import ping
from multiprocessing import Pool


def func(i):
    print("start ping ", i)
    while True:
        ping("192.168.0.1")


if __name__ == "__main__":
    pool_num = 2

    pool = Pool(processes=pool_num)
    process_list = []
    for i in range(pool_num):
        p = pool.apply_async(func=func, args=(i,))
        process_list.append(p)

    pool.close()
    pool.join()
