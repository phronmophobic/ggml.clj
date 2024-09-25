(ns com.phronemophobic.ggml.impl.raw
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.pprint :refer [pprint]]
            [clojure.edn :as edn]
            [com.phronemophobic.clong.gen.dtype-next :as gen]
            [tech.v3.datatype.struct :as dt-struct]
            [tech.v3.datatype.protocols :as dtype-proto]
            [tech.v3.datatype.native-buffer :as native-buffer]
            [tech.v3.datatype :as dtype]
            [tech.v3.datatype.ffi :as dt-ffi])
  (:import
   [tech.v3.datatype.ffi Pointer])
  (:gen-class))


(defn long->pointer ^Pointer [n]
  (Pointer. n))

(defn ^:private write-edn [w obj]
  (binding [*print-length* nil
            *print-level* nil
            *print-dup* false
            *print-meta* false
            *print-readably* true

            ;; namespaced maps not part of edn spec
            *print-namespace-maps* false

            *out* w]
    (pr obj)))

(defn ^:private  parse-api []
  (let [default-arguments
        @(requiring-resolve 'com.phronemophobic.clong.clang/default-arguments)

        include-dir (io/file ".." "ggml" "include")

        header-files (->> include-dir
                          (.listFiles)
                          (filter #(str/ends-with? (.getName %) ".h"))
                          (map #(.getCanonicalPath %)))
        
        clang-args (conj default-arguments
                         (str "-I" (.getCanonicalPath include-dir))
                         (first header-files))
        clang-args (into clang-args
                         (mapcat (fn [h]
                                   ["-include" h]))
                         (rest header-files))
        ]
    ((requiring-resolve 'com.phronemophobic.clong.clang/easy-api) nil
     clang-args)))

(defn filter-api [api]
  (-> api
      (update :structs
              (fn [structs]
                (filterv (fn [struct]
                           (let [struct-name (-> struct :id name)]
                             (some (fn [prefix]
                                     (str/starts-with? struct-name prefix))
                                   ["ggml_"
                                    ;; anonymous structs
                                    "Struct_"])))
                         structs)))
      (update :functions
              (fn [functions]
                (->> functions
                     (remove #(#{;; these functions take or return arrays,
                                 ;; which isn't directly supported
                                 "ggml_backend_guid"
                                 "ggml_guid_matches"
                                 "ggml_rope_yarn_corr_dims"} (:symbol %) ))
                     (filterv #(str/starts-with? (:symbol %) "ggml_")))))))

(defn dump-api []
  (let [outf (io/file
              "resources"
              "com"
              "phronemophobic"
              "ggml"
              "api.edn")]
    (.mkdirs (.getParentFile outf))
    (with-open [w (io/writer outf)]
      (write-edn w
                 (-> (parse-api)
                     (filter-api))))))

(defn load-api []
  (with-open [rdr (io/reader
                   (io/resource
                    "com/phronemophobic/ggml/api.edn"))
              rdr (java.io.PushbackReader. rdr)]
    (edn/read rdr)))


(defn ptr-array-fields-workaround [structs]
  (into []
        (map (fn [[id fields]]
               [id (into []
                         (map (fn [field]
                                (if (and (= :pointer (:datatype field))
                                         (> (:n-elems field)
                                            1))
                                  (assoc field :datatype :int64)
                                  field)))
                         fields)]))
        structs))

(def banned
  #{:ggml_backend_rpc_buffer_type
     :ggml_vk_instance_init
     :ggml_backend_vk_get_device_description
     :ggml_vk_get_device
     :ggml_backend_cuda_get_device_memory
     :ggml_sycl_get_gpu_list
     :ggml_backend_cuda_init
     :ggml_backend_cuda_unregister_host_buffer
     :ggml_backend_cuda_host_buffer_type
     :ggml_backend_sycl_split_buffer_type
     :ggml_backend_cuda_get_device_count
     :ggml_backend_sycl_buffer_type
     :ggml_backend_kompute_buffer_type
     :ggml_backend_kompute_init
     :ggml_backend_vk_host_buffer_type
     :ggml_backend_cuda_buffer_type
     :ggml_backend_cuda_split_buffer_type
     :ggml_backend_vk_get_device_count
     :ggml_sycl_get_device_description
     :ggml_backend_is_vk
     :ggml_vk_current_device
     :ggml_backend_rpc_get_device_memory
     :ggml_backend_sycl_get_device_memory
     :ggml_backend_sycl_print_sycl_devices
     :ggml_backend_rpc_init
     :ggml_backend_cuda_register_host_buffer
     :ggml_backend_cuda_log_set_callback
     :ggml_backend_is_kompute
     :ggml_backend_vk_buffer_type
     :ggml_backend_sycl_init
     :ggml_vk_has_vulkan
     :ggml_backend_cuda_get_device_description
     :ggml_backend_vk_init
     :ggml_vk_has_device
     :ggml_backend_is_rpc
     :ggml_backend_sycl_get_device_count
     :ggml_vk_available_devices
     :ggml_backend_sycl_host_buffer_type
     :ggml_backend_vk_get_device_memory
     :ggml_backend_is_cuda})

(defn filter-fns [interface]
  (into {}
        (comp (remove (fn [[id info]]
                        (#{ ;; these functions take or return arrays,
                           ;; which isn't directly supported
                           "ggml_backend_guid"
                           "ggml_guid_matches"
                           "ggml_rope_yarn_corr_dims"} (name id))))
              (remove (fn [[id info]]
                        (banned id))))
        interface))

(def api
  (load-api))

(gen/def-enums api)

(def dtype-structs (-> (gen/api->structs api)
                       ptr-array-fields-workaround))
(doseq [[id fields] dtype-structs]
  (dt-struct/define-datatype! id fields))


(def dtype-interface (-> (gen/api->library-interface api)
                         filter-fns))
(defmacro chunk-define []
  `(do
     ~@(into
        []
        (comp (map-indexed
               (fn [i chunk]
                 (let [interface (into {} chunk)
                       classname (symbol (str (ns-name *ns* ) (str ".Bindings" (gensym) i)))]
                   `(dt-ffi/define-library-interface (quote ~interface)
                      :classname (quote ~classname)
                      :libraries ["ggml"]
                      )))))
        (partition-all 5 dtype-interface))))
;; Works around a "Method code too large" exception
;; if all the functions are defined at once
(chunk-define)


