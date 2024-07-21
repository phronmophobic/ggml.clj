(ns com.phronemophobic.ggml.impl.raw
  (:require [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.pprint :refer [pprint]]
            [clojure.edn :as edn]
            [com.phronemophobic.clong.gen.jna :as gen])
  (:import
   java.io.PushbackReader
   com.sun.jna.Memory
   com.sun.jna.Pointer
   com.sun.jna.ptr.PointerByReference
   com.sun.jna.ptr.LongByReference
   com.sun.jna.Structure)
  (:gen-class))

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
  (update api :functions
          (fn [functions]
            (filterv #(str/starts-with? (:symbol %) "ggml_")
                     functions))))

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



(def lib
  (delay
    (com.sun.jna.NativeLibrary/getInstance "ggml")))

(def api
  (load-api)
  )

(gen/def-api-lazy lib api)


(let [struct-prefix (gen/ns-struct-prefix *ns*)]
  (defmacro import-structs []
    `(gen/import-structs! api ~struct-prefix)))
