import os
import shutil


def main():
    rootpath = "E:\\speech\\dataset\\IEMOCAP语料库"
    targetPath = rootpath + "\\wav2"
    for i in range(5):
        SessNum = i + 1
        SessionPath = rootpath + "\\Session" + str(SessNum)
        wavPath = SessionPath + "\\sentences\\wav"
        for root, dirs, files in os.walk(wavPath):
            j = 0
            for file in files:
                j = j + 1
                # print(j)
                src_file = os.path.join(root, file)
                print("src_file:" + src_file)
                shutil.copy(src_file, targetPath)


if __name__ == '__main__':
    main()
