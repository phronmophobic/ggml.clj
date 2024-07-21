(ns com.phronemophobic.ggml.simple
  (:require [clojure.java.io :as io]
            [clojure.pprint :refer [pprint]]
            [com.phronemophobic.ggml.impl.raw :as raw])
  (:import
   com.sun.jna.Memory
   com.sun.jna.Pointer
   com.sun.jna.ptr.PointerByReference
   com.sun.jna.ptr.LongByReference
   com.sun.jna.Structure)
  (:gen-class))

(raw/import-structs)
(defn float-buf [xs]
  (let [arr (float-array xs)
        mem (Memory. (* 4 (alength arr)))]
    (.write mem 0 arr 0 (alength arr))
    mem))

(def graph-buf (Memory. (* 10 1024 1024)))
(defn build-graph [t1 t2]
  (let [params (doto (ggml_init_params. )
                 ;; 10M
                 (.writeField "mem_size" (.size graph-buf))
                 (.writeField "mem_buffer" graph-buf)
                 (.writeField "no_alloc" (byte 1)))
        ctx (raw/ggml_init params)
        gf (raw/ggml_new_graph ctx)
        result (raw/ggml_mul_mat ctx t1 t2)]
    (raw/ggml_build_forward_expand gf result)
    (raw/ggml_free ctx)
    gf))

(defn compute [backend t1 t2 allocr]
  (let [gf (build-graph t1 t2)]
    (raw/ggml_gallocr_alloc_graph allocr gf)

    (raw/ggml_backend_cpu_set_n_threads backend 1)

    (raw/ggml_backend_graph_compute backend gf)

    (let [
          n-nodes (.readField gf "n_nodes")
          nodes (.getPointerArray (.readField gf "nodes") 0 n-nodes)]
      (last nodes))))

(defn -main []

  (def backend (raw/ggml_backend_cpu_init))

  (def init-params
    (doto (ggml_init_params. )
      ;; 10M
      (.writeField "mem_size" (* 10 1024 1024))
      (.writeField "no_alloc" (byte 1))
      ))

  (def ctx (raw/ggml_init init-params))

  (def t1 (raw/ggml_new_tensor_2d ctx raw/GGML_TYPE_F32 2 4))
  (def t2 (raw/ggml_new_tensor_2d ctx raw/GGML_TYPE_F32 2 3))

  (def buffer (raw/ggml_backend_alloc_ctx_tensors ctx backend))

  
  (let [nums [2 8
              5 1
              4 2
              8 6]]
    (raw/ggml_backend_tensor_set
     t1
     (float-buf nums)
     0 (* 4 (count nums))))

  (let [nums [10 5
              9 9
              5 4]]
    (raw/ggml_backend_tensor_set
     t2
     (float-buf nums)
     0 (* 4 (count nums))))

  (def gf (build-graph t1 t2))

  (def allocr
    (let [allocr (raw/ggml_gallocr_new (raw/ggml_backend_get_default_buffer_type backend))

          gf (build-graph t1 t2)
          _ (raw/ggml_gallocr_reserve allocr gf)
          mem-size (raw/ggml_gallocr_get_buffer_size allocr 0)]
      (prn (/ mem-size 1024.0))
      allocr))

  (def result-node (compute backend t1 t2 allocr))

  (def result-n (raw/ggml_nelements result-node))
  (def result-out (Memory. (* 4 result-n)))
  (raw/ggml_backend_tensor_get result-node result-out 0 (raw/ggml_nbytes result-node))

  (prn (seq (.getFloatArray result-out 0 result-n)))

  
  

  

  

  )
