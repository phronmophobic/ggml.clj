(ns com.phronemophobic.ggml.makemore
  (:require [com.phronemophobic.ggml :as ggml]
            [com.phronemophobic.ggml.impl.raw :as raw]
            [tech.v3.datatype.struct :as dt-struct]
            [tech.v3.datatype.protocols :as dtype-proto]
            [tech.v3.datatype.native-buffer :as native-buffer]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.ffi :as dt-ffi]
            [tech.v3.tensor :as dtt]
            [clojure.string :as str]
            [clojure.java.io :as io]))

(def words (str/split-lines
            (slurp (io/resource "names.txt"))))


;; https://github.com/karpathy/nn-zero-to-hero/blob/master/lectures/makemore/makemore_part2_mlp.ipynb

;; import torch
;; import torch.nn.functional as F
;; import matplotlib.pyplot as plt # for making figures
;; %matplotlib inline

;; # read in all the words
;; words = open('names.txt', 'r').read().splitlines()
;; words[:8]

;; len(words)

;; # build the vocabulary of characters and mappings to/from integers
;; chars = sorted(list(set(''.join(words))))
;; stoi = {s:i+1 for i,s in enumerate(chars)}
;; stoi['.'] = 0
;; itos = {i:s for s,i in stoi.items()}
;; print(itos)

;; # build the dataset
(def chars (->> words
                (apply concat)
                distinct
                sort
                vec))
(def stoi (-> (into {}
                    (map-indexed (fn [i c]
                                   [c (inc i)]))
                    chars)
              (assoc "." 0)))
(def itos (into {}
                (map (fn [[k v]]
                       [v k]))
                stoi))

;; block_size = 3 # context length: how many characters do we take to predict the next one?

(def block-size 3)
;; X, Y = [], []
;; for w in words:
  
;;   #print(w)
;;   context = [0] * block_size
;;   for ch in w + '.':
;;     ix = stoi[ch]
;;     X.append(context)
;;     Y.append(ix)
;;     #print(''.join(itos[i] for i in context), '--->', itos[ix])
;;     context = context[1:] + [ix] # crop and append


(def YX
  (into []
        (comp
         (take 10)
         (mapcat
          (fn [word]
            (eduction
             ;; tuple of [next-char vector-of-block-size-preceding-chars]
             (map (fn [i]
                    [(nth word i ".")
                     (into []
                           (comp
                            (map (fn [j]
                                   (nth word j "."))))
                           (range (- i block-size) i))]))
             ;; map characters to numbers
             (map (fn [[k context]]
                    [(stoi k)
                     (mapv stoi context)]))
             (range (+ (count word)
                       1))))))
        words))

(def X (-> (dtt/->tensor (into
                       []
                       (mapcat second)
                       YX)
                      :datatype :int32
                      :container-type :native-heap)))
(def Y (dtt/->tensor (into
                      []
                      (map first)
                      YX)
                     :datatype :int32
                     :container-type :native-heap))


(def ^:private ggml-type->dtype
  { ;; raw/GGML_TYPE_BF16
   ;; raw/GGML_TYPE_F16
   raw/GGML_TYPE_F32 :float32
   raw/GGML_TYPE_F64 :float64
   raw/GGML_TYPE_I16 :int16
   raw/GGML_TYPE_I32 :int32
   raw/GGML_TYPE_I64 :int64
   raw/GGML_TYPE_I8 :int8
   ;;  raw/GGML_TYPE_IQ1_M
   ;;  raw/GGML_TYPE_IQ1_S
   ;;  raw/GGML_TYPE_IQ2_S
   ;;  raw/GGML_TYPE_IQ2_XS
   ;;  raw/GGML_TYPE_IQ2_XXS
   ;;  raw/GGML_TYPE_IQ3_S
   ;;  raw/GGML_TYPE_IQ3_XXS
   ;;  raw/GGML_TYPE_IQ4_NL
   ;;  raw/GGML_TYPE_IQ4_XS
   ;;  raw/GGML_TYPE_Q2_K
   ;;  raw/GGML_TYPE_Q3_K
   ;;  raw/GGML_TYPE_Q4_0
   ;;  raw/GGML_TYPE_Q4_1
   ;;  raw/GGML_TYPE_Q4_K
   ;;  raw/GGML_TYPE_Q5_0
   ;;  raw/GGML_TYPE_Q5_1
   ;;  raw/GGML_TYPE_Q5_K
   ;;  raw/GGML_TYPE_Q6_K
   ;;  raw/GGML_TYPE_Q8_0
   ;;  raw/GGML_TYPE_Q8_1
   ;; raw/GGML_TYPE_Q8_K
   }
  )
(def ^:private dtype->ggml-type
  (into {}
        (map (fn [[a b]]
               [b a]))
        ggml-type->dtype))

;; (def params (dt-struct/map->struct :ggml_init_params
;;                                    {:mem_size (* 16 1024 1024)
;;                                     :no_alloc 1}))
;; (defonce ctx (raw/ggml_init params))
(def emb-size 2)
(def num-tokens 27)
;; (def C (raw/ggml_new_tensor_2d ctx (dtype->ggml-type :float32) emb-size num-tokens))
;; C = torch.randn((27, 2))
(def C (dtt/new-tensor [27 emb-size]
                       :datatype :float32
                       :container-type :native-heap))
(dtype/copy! 
 (dtt/compute-tensor
  [27 emb-size]
  (fn [& args]
    (rand))
  :float32)
 C)


;; X = torch.tensor(X)

;; Y = torch.tensor(Y)

;; X.shape, X.dtype, Y.shape, Y.dtype


;; # build the dataset
;; block_size = 3 # context length: how many characters do we take to predict the next one?

;; def build_dataset(words):  
;;   X, Y = [], []
;;   for w in words:

;;     #print(w)
;;     context = [0] * block_size
;;     for ch in w + '.':
;;       ix = stoi[ch]
;;       X.append(context)
;;       Y.append(ix)
;;       #print(''.join(itos[i] for i in context), '--->', itos[ix])
;;       context = context[1:] + [ix] # crop and append

;;   X = torch.tensor(X)
;;   Y = torch.tensor(Y)
;;   print(X.shape, Y.shape)
;;   return X, Y

;; import random
;; random.seed(42)
;; random.shuffle(words)
;; n1 = int(0.8*len(words))
;; n2 = int(0.9*len(words))

;; Xtr, Ytr = build_dataset(words[:n1])
;; Xdev, Ydev = build_dataset(words[n1:n2])
;; Xte, Yte = build_dataset(words[n2:])

;; torch.Size([182441, 3]) torch.Size([182441])
;; torch.Size([22902, 3]) torch.Size([22902])
;; torch.Size([22803, 3]) torch.Size([22803])

(def my-graph
  (fn [ctx X C]
    (let [ ;; emb = C[X]
          flat-emb (raw/ggml_get_rows ctx C X)
          emb (raw/ggml_reshape_3d ctx flat-emb emb-size block-size (count YX))
          loss (raw/ggml_sum ctx emb)]
      [emb loss])))
;; C = torch.randn((27, 2))




;; emb.shape



;; W1 = torch.randn((6, 100))
;; b1 = torch.randn(100)

;; h = torch.tanh(emb.view(-1, 6) @ W1 + b1)

;; h

;; h.shape

;; W2 = torch.randn((100, 27))
;; b2 = torch.randn(27)

;; logits = h @ W2 + b2

;; logits.shape

;; counts = logits.exp()

;; prob = counts / counts.sum(1, keepdims=True)

;; prob.shape

;; loss = -prob[torch.arange(32), Y].log().mean()
;; loss

;; # ------------ now made respectable :) ---------------

;; Xtr.shape, Ytr.shape # dataset

;; (torch.Size([182441, 3]), torch.Size([182441]))

;; g = torch.Generator().manual_seed(2147483647) # for reproducibility
;; C = torch.randn((27, 10), generator=g)
;; W1 = torch.randn((30, 200), generator=g)
;; b1 = torch.randn(200, generator=g)
;; W2 = torch.randn((200, 27), generator=g)
;; b2 = torch.randn(27, generator=g)
;; parameters = [C, W1, b1, W2, b2]

;; sum(p.nelement() for p in parameters) # number of parameters in total

;; 11897

;; for p in parameters:
;;   p.requires_grad = True

;; lre = torch.linspace(-3, 0, 1000)
;; lrs = 10**lre

;; lri = []
;; lossi = []
;; stepi = []

;; for i in range(200000):
  
;;   # minibatch construct
;;   ix = torch.randint(0, Xtr.shape[0], (32,))
  
;;   # forward pass
;;   emb = C[Xtr[ix]] # (32, 3, 10)
;;   h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 200)
;;   logits = h @ W2 + b2 # (32, 27)
;;   loss = F.cross_entropy(logits, Ytr[ix])
;;   #print(loss.item())
  
;;   # backward pass
;;   for p in parameters:
;;     p.grad = None
;;   loss.backward()
  
;;   # update
;;   #lr = lrs[i]
;;   lr = 0.1 if i < 100000 else 0.01
;;   for p in parameters:
;;     p.data += -lr * p.grad

;;   # track stats
;;   #lri.append(lre[i])
;;   stepi.append(i)
;;   lossi.append(loss.log10().item())

;; #print(loss.item())

;; plt.plot(stepi, lossi)

;; [<matplotlib.lines.Line2D at 0x7feda5f10250>]

;; emb = C[Xtr] # (32, 3, 2)
;; h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
;; logits = h @ W2 + b2 # (32, 27)
;; loss = F.cross_entropy(logits, Ytr)
;; loss

;; tensor(2.1260, grad_fn=<NllLossBackward0>)

;; emb = C[Xdev] # (32, 3, 2)
;; h = torch.tanh(emb.view(-1, 30) @ W1 + b1) # (32, 100)
;; logits = h @ W2 + b2 # (32, 27)
;; loss = F.cross_entropy(logits, Ydev)
;; loss

;; tensor(2.1701, grad_fn=<NllLossBackward0>)

;; # visualize dimensions 0 and 1 of the embedding matrix C for all characters
;; plt.figure(figsize=(8,8))
;; plt.scatter(C[:,0].data, C[:,1].data, s=200)
;; for i in range(C.shape[0]):
;;     plt.text(C[i,0].item(), C[i,1].item(), itos[i], ha="center", va="center", color='white')
;; plt.grid('minor')

;; # training split, dev/validation split, test split
;; # 80%, 10%, 10%

;; context = [0] * block_size
;; C[torch.tensor([context])].shape

;; torch.Size([1, 3, 10])

;; # sample from the model
;; g = torch.Generator().manual_seed(2147483647 + 10)

;; for _ in range(20):
    
;;     out = []
;;     context = [0] * block_size # initialize with all ...
;;     while True:
;;       emb = C[torch.tensor([context])] # (1,block_size,d)
;;       h = torch.tanh(emb.view(1, -1) @ W1 + b1)
;;       logits = h @ W2 + b2
;;       probs = F.softmax(logits, dim=1)
;;       ix = torch.multinomial(probs, num_samples=1, generator=g).item()
;;       context = context[1:] + [ix]
;;       out.append(ix)
;;       if ix == 0:
;;         break
    
;;     print(''.join(itos[i] for i in out))


(defn -main []
  
(let [cpu-sched (ggml/cpu-scheduler)]

  (prn (ggml/compute cpu-sched
                       my-graph
                       X C))
  (identity cpu-sched))
  
