(ns com.phronemophobic.ggml
  (:require [clojure.java.io :as io]
            [clojure.pprint :refer [pprint]]
            [com.phronemophobic.ggml.impl.raw :as raw]
            [tech.v3.datatype.struct :as dt-struct]
            [tech.v3.datatype.protocols :as dtype-proto]
            [tech.v3.datatype.native-buffer :as native-buffer]
            [tech.v3.datatype :as dtype]
            [tech.v3.tensor :as dtt]
            [tech.v3.datatype.ffi :as dt-ffi])
  (:import [tech.v3.datatype.ffi Pointer])
  (:gen-class))

(defn ^:private ptr->struct [^Pointer ptr dtype]
  (let [buf (native-buffer/wrap-address (.address ptr)
                                        (-> dtype
                                            dt-struct/get-struct-def
                                            :datatype-size)
                                          nil)]
     (dt-struct/inplace-new-struct
              dtype
              buf)))

;; Only :float32 seems to be widely supported
;; for most ops?
(def ^:private ggml-type->dtype
  {;; raw/GGML_TYPE_BF16
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


(defn tensor-dtype [ptr]
  (let [tensor (ptr->struct ptr :ggml_tensor)
        dtype (ggml-type->dtype (:type tensor))]
    (assert dtype)
    dtype))

(defn buffer-ggml-type [buf]
  (let [ggml-type (get dtype->ggml-type (dtype-proto/elemwise-datatype buf))]
    (assert ggml-type)
    ggml-type))


(defn my-graph [ctx a b]
  (let [out (raw/ggml_scale ctx (raw/ggml_add ctx a b) -1)]
    [out
     (raw/ggml_sum ctx out)
     (raw/ggml_sum_rows ctx out)]))

(def log-callback-iface (dt-ffi/define-foreign-interface :void [:int32 :pointer :pointer]))
(def log-callback-instance
  (dt-ffi/instantiate-foreign-interface
   log-callback-iface
   (fn [log-level msg _]
     (print (dt-ffi/c->string msg)))))
(def log-callback-ptr (dt-ffi/foreign-interface-instance->c log-callback-iface log-callback-instance))

(defn gpu-scheduler []

  (let [
        cpu-backend (raw/ggml_backend_cpu_init)

        _ (raw/ggml_backend_metal_log_set_callback
           log-callback-ptr
           nil)
        metal-backend (doto (raw/ggml_backend_metal_init)
                        (raw/ggml_backend_metal_set_n_cb 4))

        backends [metal-backend cpu-backend ]
        backends-buf (dtype/make-container :native-heap
                                           :int64
                                           (into []
                                                 (map #(.address ^Pointer %))
                                                 backends))
        #_(let [mem (Memory. (* 8 (count backends)))]
            (.write mem 0 (into-array Pointer backends) 0 (count backends))
            mem)
        parallel? false

        sched (raw/ggml_backend_sched_new backends-buf nil (count backends) 2048 (if parallel? 1 0))]
    sched))

(defn cpu-scheduler []

  (let [
        cpu-backend (raw/ggml_backend_cpu_init)

        _ (raw/ggml_backend_metal_log_set_callback
           log-callback-ptr
           nil)
        metal-backend (doto (raw/ggml_backend_metal_init)
                        (raw/ggml_backend_metal_set_n_cb 4))

        backends [ cpu-backend ]
        backends-buf (dtype/make-container :native-heap
                                           :int64
                                           (into []
                                                 (map #(.address ^Pointer %))
                                                 backends))
        parallel? false

        sched (raw/ggml_backend_sched_new backends-buf nil (count backends) 2048 (if parallel? 1 0))]
    sched))


(defn ->tensor
  "Constructs a ggml tensor from a dtype tensor"
  [ctx dtt]
  (let [tensor (raw/ggml_new_tensor ctx
                                    (buffer-ggml-type dtt)
                                    (.rank dtt)
                                    (dtype/make-container :native-heap :int64
                                                          (dtype/shape dtt)))]
    (let [buf (native-buffer/ensure-native dtt)]
      (raw/ggml_backend_tensor_set tensor buf 0 (native-buffer/native-buffer-byte-len buf)))))

(defn get-shape [tensor]
  (let [struct (ptr->struct tensor :ggml_tensor)
        shape (:ne struct)]
    ;; remove degenerate higher dimensions
    ;;     every tensor is technically 4 dimensions with
    ;;     for eg, a 2d vector will have the 3rd and 4th
    ;;     dimension set to 1
    (if (= shape [1 1 1 1])
      [1]
      (->> shape
           reverse
           (drop-while #(= 1 %))
           reverse
           (into [])))))

(defn compute [scheduler f & inputs]

  (let [params
        (dt-struct/map->struct :ggml_init_params
                               {:mem_size (* 16 1024 1024)
                                :no_alloc 1})
        #_(doto (ggml_init_params. )
          ;; 10M
          (.writeField "mem_size" (* 16 1024 1024))
          (.writeField "no_alloc" (byte 1)))
        ctx (raw/ggml_init params)
        tensors (mapv (fn [arr]
                        (let [dshape (dtype/shape arr)]
                          (raw/ggml_new_tensor ctx
                                               (buffer-ggml-type arr)
                                               (count dshape)
                                               (dtype/make-container :native-heap :int64
                                                                     (reverse dshape)))))
                      inputs)

        gf (raw/ggml_new_graph ctx)
        _ (assert gf)
        outputs (apply f ctx tensors)
        _ (doseq [output outputs]
            (raw/ggml_build_forward_expand gf output))

        _ (do
            (raw/ggml_backend_sched_reserve scheduler gf)
            (raw/ggml_backend_sched_reset scheduler)
            (raw/ggml_backend_sched_alloc_graph scheduler gf))
        _ (doseq [[arr tensor] (map vector inputs tensors)]
            (let [buf (native-buffer/ensure-native arr)]
              (raw/ggml_backend_tensor_set tensor buf 0 (native-buffer/native-buffer-byte-len buf))))

        _   (raw/ggml_backend_sched_graph_compute scheduler gf)

        results (into []
                      (map (fn [output]
                             (let [shape (reverse (get-shape output))
                                   dtype (tensor-dtype output)
                                   dtt (dtt/native-tensor shape dtype)
                                   buf (native-buffer/ensure-native dtt)]
                               (raw/ggml_backend_tensor_get output buf 0 (native-buffer/native-buffer-byte-len buf))
                               dtt)))
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
         (raw/ggml_sum ctx out)
         #_(raw/ggml_sum_rows ctx out)])))


  (def n 10000000)
  ;; (def a (float-array (repeatedly n rand)))
  ;; (def b (float-array (repeatedly n rand)))

  (def a (dtt/->tensor (repeat 3 (range 2)) :container-type :native-heap :datatype :float32))
  (def b (dtt/->tensor (repeat 3 (range 2)) :container-type :native-heap :datatype :float32))

  ;; (def a (dtype/make-container :native-heap :float32 (range 10)))
  ;; (def b (dtype/make-container :native-heap :float32 (range 10)))


  (def result-cpu (time (compute cpu-sched my-graph a b)))
  (def result-gpu (time (compute gpu-sched my-graph a b)))

  (prn result-cpu
       result-gpu)

  ,)

