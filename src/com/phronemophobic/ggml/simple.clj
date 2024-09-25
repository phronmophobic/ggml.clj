(ns com.phronemophobic.ggml.simple
  (:require [clojure.java.io :as io]
            [clojure.pprint :refer [pprint]]
            [com.phronemophobic.ggml.impl.raw :as raw]
            [tech.v3.datatype.struct :as dt-struct]
            [tech.v3.datatype.protocols :as dtype-proto]
            [tech.v3.datatype.native-buffer :as native-buffer]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.ffi :as dt-ffi])
  (:import [tech.v3.datatype.ffi Pointer])
  (:gen-class))

(def graph-buf
  (native-buffer/malloc (* 10 1024 1024)))
(defn build-graph [t1 t2]
  (let [;; params (doto (ggml_init_params. )
        ;;          ;; 10M
        ;;          (.writeField "mem_size" (.size graph-buf))
        ;;          (.writeField "mem_buffer" graph-buf)
        ;;          (.writeField "no_alloc" (byte 1)))
        params (dt-struct/map->struct :ggml_init_params
                                      {:mem_size (native-buffer/native-buffer-byte-len graph-buf)
                                       :mem_buffer (.address (dt-ffi/->pointer graph-buf))
                                       :no_alloc 1})
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
          buf (native-buffer/wrap-address (.address gf)
                                          (-> :ggml_cgraph
                                              dt-struct/get-struct-def
                                              :datatype-size)
                                          nil)
          gf (dt-struct/inplace-new-struct
              :ggml_cgraph
              buf)

          n-nodes (:n_nodes gf) #_(.readField gf "n_nodes")

          nodes (-> (native-buffer/wrap-address (:nodes gf)
                                             (* 8 n-nodes)
                                             nil)
                    (native-buffer/set-native-datatype :int64))
          last-node-address (last nodes)
          #_(ffi/long->pointer (.getLong  (native-buffer/unsafe) (:nodes gf)))

          ;;nodes (.getPointerArray (.readField gf "nodes") 0 n-nodes)
          ]
      (Pointer. last-node-address))))

(defn -main []

  (def backend (raw/ggml_backend_cpu_init))


  (def init-params
    (dt-struct/map->struct :ggml_init_params
                           {:mem_size (* 10 1024 1024)
                            :no_alloc 1})
    #_(doto (ggml_init_params. )
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
     ;;(float-buf nums)
     (dtype/make-container :native-heap :float32 nums)
     0 (* 4 (count nums))))

  (let [nums [10 5
              9 9
              5 4]]
    (raw/ggml_backend_tensor_set
     t2
     (dtype/make-container :native-heap :float32 nums)
     0 (* 4 (count nums))))

  (def gf (build-graph t1 t2))

  (def allocr
    (let [allocr (raw/ggml_gallocr_new (raw/ggml_backend_get_default_buffer_type backend))

          gf (build-graph t1 t2)
          _ (raw/ggml_gallocr_reserve allocr gf)
          mem-size (raw/ggml_gallocr_get_buffer_size allocr 0)]
      allocr))

  (def result-node (compute backend t1 t2 allocr))

  (def result-n (raw/ggml_nelements result-node))
  
  ;; (def result-out (Memory. (* 4 result-n)))
  (def result-out (dtype/make-container :native-heap :float32 result-n))
  (raw/ggml_backend_tensor_get result-node result-out 0 (raw/ggml_nbytes result-node))

  (prn result-out)
  ;; (prn (seq (.getFloatArray result-out 0 result-n)))

  
  

  

  

  )
