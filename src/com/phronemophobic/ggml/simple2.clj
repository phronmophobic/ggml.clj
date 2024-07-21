(ns com.phronemophobic.ggml.simple2
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


(defn -main []

  (def params
    (doto (ggml_init_params. )
      ;; 10M
      (.writeField "mem_size" (* 16 1024 1024))
      (.writeField "no_alloc" (byte 1))
      
      ))
  ;; setup graph

  (def ctx (raw/ggml_init params))

  ;; (def x (raw/ggml_new_tensor_1d ctx raw/GGML_TYPE_F32 1000))
  ;; (raw/ggml_set_param  ctx x)

  ;; (def a (raw/ggml_new_tensor_1d ctx raw/GGML_TYPE_F32 1))
  ;; (def b (raw/ggml_new_tensor_1d ctx raw/GGML_TYPE_F32 1))
  ;; (def x2 (raw/ggml_mul ctx x x))
  ;; (def f (raw/ggml_add ctx (raw/ggml_mul ctx a x2) b))

  (def a (raw/ggml_new_tensor_1d ctx raw/GGML_TYPE_F32 1000))
  (def b (raw/ggml_new_tensor_1d ctx raw/GGML_TYPE_F32 1000))



  (def output1 (raw/ggml_mul ctx
                            (raw/ggml_add ctx a b)
                            (raw/ggml_sub ctx a b)))

  (def output2 (raw/ggml_abs ctx output1))

  (def gf (raw/ggml_new_graph ctx))
  (raw/ggml_build_forward_expand gf output1)
  (raw/ggml_build_forward_expand gf output2)


  ;; setup scheduler

  (def cpu-backend (raw/ggml_backend_cpu_init))


  (raw/ggml_backend_metal_log_set_callback
   (fn [log-level msg _]
     (prn (.getString (.getPointer msg) 0)))
   nil)
  (def metal-backend (raw/ggml_backend_metal_init))
  (println metal-backend)
  (raw/ggml_backend_metal_set_n_cb metal-backend 4)

  (def backends [metal-backend cpu-backend ])
  (def backends-buf (let [mem (Memory. (* 8 (count backends)))]
                      (.write mem 0 (into-array Pointer backends) 0 (count backends))
                      mem))
  (def parallel? false)

  (def sched (raw/ggml_backend_sched_new backends-buf nil (count backends) 2048 (if parallel? 1 0)))



  ;; alloc tensor buffers

  ;; optionally, set tensor to backend
  ;; (doseq [tensor [a b c]]
  ;;   (raw/ggml_backend_sched_set_tensor_backend sched tensor metal-backend))

  (raw/ggml_backend_sched_reserve sched gf)
  (raw/ggml_backend_sched_reset sched)
  (raw/ggml_backend_sched_alloc_graph sched gf)

  ;; run

  ;; (raw/ggml_set  )

  (let [data (Memory. (* 4 1000))]
    (.write data 0 (float-array (repeatedly 1000 (constantly 42))) 0 1000)
    (raw/ggml_backend_tensor_set a data 0 (.size data)))

  (let [data (Memory. (* 4 1000))]
    (.write data 0 (float-array (repeatedly 1000 (constantly 10))) 0 1000)
    (raw/ggml_backend_tensor_set b data 0 (.size data)))

  ;; (raw/ggml_set_f32 x 2.0)
  ;; (raw/ggml_set_f32 a 3.0)
  ;; (raw/ggml_set_f32 b 5.0)

  

  #_(raw/ggml_graph_compute_with_ctx ctx (PointerByReference. (.getPointer gf)) 1)

  (raw/ggml_backend_sched_graph_compute sched gf)
  



  (def result-n (raw/ggml_nelements output1))
  (def result-out (Memory. (raw/ggml_nbytes output1)))
  (raw/ggml_backend_tensor_get output1 result-out 0 (.size result-out))
  (prn (seq (.getFloatArray result-out 0 result-n)))


    (def result-n (raw/ggml_nelements output2))
  (def result-out (Memory. (raw/ggml_nbytes output2)))
  (raw/ggml_backend_tensor_get output2 result-out 0 (.size result-out))
  (prn (seq (.getFloatArray result-out 0 result-n)))


  )
