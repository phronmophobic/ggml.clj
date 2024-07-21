(ns com.phronemophobic.ggml
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

(defn my-graph [ctx a b]
  (let [out (raw/ggml_scale ctx (raw/ggml_add ctx a b) -1)]
    [out
     (raw/ggml_sum_rows ctx out)]))

(defn gpu-scheduler []

  (let [
        cpu-backend (raw/ggml_backend_cpu_init)

        _ (raw/ggml_backend_metal_log_set_callback
           (fn [log-level msg _]
             ;;(prn (.getString (.getPointer msg) 0))
             )
           nil)
        metal-backend (doto (raw/ggml_backend_metal_init)
                        (raw/ggml_backend_metal_set_n_cb 4))

        backends [metal-backend cpu-backend ]
        backends-buf (let [mem (Memory. (* 8 (count backends)))]
                       (.write mem 0 (into-array Pointer backends) 0 (count backends))
                       mem)
        parallel? false

        sched (raw/ggml_backend_sched_new backends-buf nil (count backends) 2048 (if parallel? 1 0))]
    sched))

(defn cpu-scheduler []

  (let [
        cpu-backend (raw/ggml_backend_cpu_init)

        _ (raw/ggml_backend_metal_log_set_callback
           (fn [log-level msg _]
             (prn (.getString (.getPointer msg) 0)))
           nil)
        metal-backend (doto (raw/ggml_backend_metal_init)
                        (raw/ggml_backend_metal_set_n_cb 4))

        backends [ cpu-backend ]
        backends-buf (let [mem (Memory. (* 8 (count backends)))]
                       (.write mem 0 (into-array Pointer backends) 0 (count backends))
                       mem)
        parallel? false

        sched (raw/ggml_backend_sched_new backends-buf nil (count backends) 2048 (if parallel? 1 0))]
    sched))



(defn compute [scheduler f & inputs]

  (let [params (doto (ggml_init_params. )
                 ;; 10M
                 (.writeField "mem_size" (* 16 1024 1024))
                 (.writeField "no_alloc" (byte 1)))
        ctx (raw/ggml_init params)
        tensors (mapv (fn [arr]
                        (raw/ggml_new_tensor_1d ctx raw/GGML_TYPE_F32 (alength arr)))
                      inputs)

        gf (raw/ggml_new_graph ctx)
        outputs (apply f ctx tensors)
        _ (doseq [output outputs]
            (raw/ggml_build_forward_expand gf output))

        _ (do
            (raw/ggml_backend_sched_reserve scheduler gf)
            (raw/ggml_backend_sched_reset scheduler)
            (raw/ggml_backend_sched_alloc_graph scheduler gf))
        _ (doseq [[arr tensor] (map vector inputs tensors)]
            (let [data (Memory. (raw/ggml_nbytes tensor))]
              (.write data 0 arr 0 (alength arr))
              (raw/ggml_backend_tensor_set tensor data 0 (.size data))))

        _   (raw/ggml_backend_sched_graph_compute scheduler gf)

        results (into []
                      (map (fn [output]
                             (let [n (raw/ggml_nelements output)
                                   buf (Memory. (raw/ggml_nbytes output))
                                   _ (raw/ggml_backend_tensor_get output buf 0 (.size buf))]
                               (.getFloatArray buf 0 n))))
                      outputs)]
    results))


(defn -main []

  (def cpu-sched (cpu-scheduler))
  (def gpu-sched (gpu-scheduler))

  (def my-graph
    (fn my-graph [ctx a b]
      (let [out (raw/ggml_scale ctx (raw/ggml_add ctx a b) -1)]
        ;; multiple outputs
        [out
         (raw/ggml_sum_rows ctx out)])))

  (def n 10000)
  (def a (float-array (repeatedly n rand)))
  (def b (float-array (repeatedly n rand)))

  (def result-cpu (time (compute cpu-sched my-graph a b)))
  (def result-gpu (time (compute gpu-sched my-graph a b)))

  (prn (-> result-cpu second seq)
       (-> result-gpu second seq))

  ,)


