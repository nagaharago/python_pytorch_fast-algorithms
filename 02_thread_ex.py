def function():
    pass


import threading

threads = []
for _ in range(10):
    thread = threading.Thread(target=function)  # スレッドを作成
    threads.append(thread)


for thread in threads:
    thread.start()  # スレッドを開始


for thread in threads:
    thread.join()  # スレッドが終了するのを待つ
