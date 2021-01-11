import crawler
import cuda_svm
import convert_img_hog

if __name__ == "__main__":
    search_category = ['apple_pie','chocolate_cake', 'donuts', 'hamburger' , 'hot_dog', 'ice_cream', 'pizza']
    count = [0,0,0,0,0,0,0]
    
    while True:
        convert_img_hog.main()
        cuda_svm.convert_to_pickele()
        acc = cuda_svm.main()
        min = acc[0]
        index = 0
        for i , acc in enumerate(acc):
            print(acc)
            if acc < min:
                index = i
        count[index]+=1
        if count[index] > 5 :
            break
        crawler.multi_thread_crawler([search_category[index]])
        
    
    