# ============================================================================
# animl-shiny
# ============================================================================
#
# Pipeline UI, built one step at a time:
#   Step 1: select a folder of camera trap images/videos, then run
#           animl::build_file_manifest() on it and preview the result.
#   Step 2: pick how many frames to pull per video, then run
#           animl::extract_frames() against that manifest.
#   Step 3: select a detector model file, then run animl::detect()
#           (via load_detector() + parse_detections()) against the
#           results so far, adding bounding boxes, confidence scores,
#           and (if a class list is provided) real category labels.
#   Step 4 (preview): click a row in the results table to view that
#           image with its bounding box drawn on it.
#
# The manifest (a table of every image/video file found, plus EXIF metadata
# like timestamps) from Step 1 is the input Step 2 needs; Step 2's output
# (or Step 1's, if there were no videos) feeds Step 3. Later pipeline
# steps -- sequence-based classification -- will build on Step 3's output
# the same way.
#
# NOTE: saving intermediate outputs to disk (out_file / out_dir args,
# WorkingDirectory()'s save-file locations) is intentionally left out of
# these helpers for now -- that's a separate concern to design properly
# later, rather than something to bolt on ad hoc here.
#
# DETECTOR MODEL_TYPE -- IMPORTANT, found by reading animl-py's own
# detect.py source directly:
#
#   animl-py's _convert_detections() silently does `category[j] += 1`
#   whenever model_type is one of the real MegaDetector variants
#   ("mdv5", "mdv6", "mdv1000-*"). This exists because real MegaDetector
#   reserves category 0 for "empty" and starts real classes at 1. A
#   CUSTOM-TRAINED model (like sdzwa_buow_detector_2026.pt) is NOT
#   MegaDetector and was never trained with that convention -- its class
#   list is 0-indexed (0=bird, 1=bobcat, 2=cattle, ...). Loading it with
#   model_type="mdv6" silently shifted every category by +1 before we
#   ever saw it.
#
#   Per load_detector()'s own docstring: "for yolo models v6+, use
#   'yolo', for v5, use 'yolovv5'". Custom (non-MegaDetector) models
#   should use "yolo" (v6+) or "yolov5" (v5) -- NOT "mdv5"/"mdv6", which
#   are reserved for genuine MegaDetector releases.
#
# CATEGORY LABELS -- detect() has a native `category_map` parameter (a
# dict of {category_id: label}) applied directly during detection.
# IMPORTANT: category_map must NEVER be NULL or a bare empty R list() --
# detect()'s own source unconditionally calls category_map.items() near
# the top of the function, crashing with "'NoneType' object has no
# attribute 'items'" on None, and an empty unnamed R list() is
# ambiguous when converted by reticulate (could become a Python list,
# which also lacks .items()). reticulate::dict() sidesteps both issues.
# For genuine MegaDetector models (mdv5/mdv6/mdv1000-*) run without a
# custom class list, default_md_category_map() supplies MD's own
# standard empty/animal/person/vehicle categories -- confirmed working
# against a real md_v5a.0.0.pt file.
#
# CONFIDENCE FILTERING -- detect() also has a native
# `confidence_threshold` parameter that drops weak detections before
# they're even returned; an image with nothing surviving that threshold
# comes back as category_label = "empty" natively.
#
# RESIZE DIMENSIONS -- MDv5/YOLOv5 requires a SQUARE input (the
# documented MegaDetector v5 standard is 1280x1280, confirmed against
# animl-py's own MEGADETECTORv5_SIZE constant). Our custom YOLO11 model
# runs fine at native resolution (2048x1440, non-square). Passing a
# non-square resize to MDv5 throws a PyTorch tensor-shape-mismatch
# error from deep inside the model -- resize dimensions are chosen
# based on model_type in the server code below, not hardcoded.
#
# VIDEO FRAME PREVIEW -- IMPORTANT, found by reading animl-py's source
# directly (get_frame_as_image() in the Python package):
#
#   extract_frames() does NOT reliably write extracted still images to
#   disk in every installation/version -- confirmed by direct testing,
#   a folder with successfully-detected video frames had zero .jpg
#   files anywhere on disk. Detection itself still works because
#   frames are read directly from the video in-memory, not from a
#   saved file. animl-py exposes this same mechanism as a public
#   function: get_frame_as_image(video_path, frame) -- uses OpenCV
#   (cv2.VideoCapture + cap.set(CAP_PROP_POS_FRAMES, frame)) to seek to
#   a specific frame and return it as an RGB array, no disk file
#   needed. Our preview calls this directly via reticulate instead of
#   guessing at an extracted-file naming convention that isn't
#   reliable. See get_video_frame_path().
#
# Required packages:
#   install.packages(c("shiny", "shinyFiles", "DT", "fs", "magick"))
#   install.packages("animl")   # or devtools::install_github("conservationtechlab/animl")
# ============================================================================

library(shiny)
library(shinyFiles)
library(DT)
library(fs)
library(magick)
library(animl)

# One-time setup: define a Python helper that reads a video frame and
# saves it to disk, entirely inside Python. This exists because letting
# the numpy array cross into R and back (R-side reference, then passed
# back to a Python call like cv2$cvtColor()) silently corrupts its
# dtype -- confirmed by direct testing, the array arrives back in
# Python as int32 instead of uint8, which OpenCV then rejects with
# "Unsupported depth of input image... depth is 4 (CV_32S)". Keeping
# the whole read -> convert -> write operation inside a single Python
# call sidesteps that conversion entirely. See get_video_frame_path().
reticulate::py_run_string("
import cv2
import animl as _animl_shiny_internal

def _animl_shiny_save_video_frame(video_path, frame, out_path):
    rgb = _animl_shiny_internal.get_frame_as_image(video_path, frame=int(frame))
    if rgb is None:
        return None
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(out_path, bgr)
    return out_path
")


# ============================================================================
# Helper functions
# ============================================================================

#' Set up animl's working directory for a folder of camera trap media
#'
#' @param imagedir string, path to a folder of camera trap images/videos
#'
#' @details WorkingDirectory() does not return an object -- it assigns
#'   global variables into whatever environment you pass it.
set_working_directory <- function(imagedir) {
  WorkingDirectory(imagedir, globalenv())
}


#' Build a file manifest for a directory of camera trap media
#'
#' @param imagedir string, path to a folder of camera trap images/videos
#'
#' @return data frame with one row per file found (FilePath, EXIF fields
#'   such as DateTimeOriginal, etc.)
build_manifest_from_dir <- function(imagedir) {
  set_working_directory(imagedir)
  build_file_manifest(imagedir, exif = TRUE)
}


#' Extract still frames from the videos in a file manifest
#'
#' @param files data frame, the manifest produced by build_manifest_from_dir()
#' @param frames_per_video integer, number of frames to sample per video
#'
#' @return data frame of still frames (one row per extracted frame)
#'
#' @details filepath stays as the ORIGINAL VIDEO path for a video's
#'   frame rows -- the frame column (e.g. 0, 100, 200) distinguishes
#'   each extracted frame. Extracted files are not reliably written to
#'   disk in every installation, so previewing/loading a video-sourced
#'   row's image uses get_video_frame_path() instead of assuming a
#'   file exists.
extract_video_frames <- function(files, frames_per_video) {
  extract_frames(
    files,
    frames      = frames_per_video,
    parallel    = TRUE,
    num_workers = parallel::detectCores()
  )
}


#' Load a detector model
#'
#' @param model_path string, path to a detector model file (.pt)
#' @param model_type string, animl model type identifier. Use "mdv5"/
#'   "mdv6" ONLY for genuine MegaDetector releases. For a custom-trained
#'   model like ours, use "yolo" (YOLO v6+) or "yolov5" (YOLO v5).
#' @param device string, compute device to load the model onto
#'
#' @return an animl detector object, ready to be passed to detect_animals()
load_md_detector <- function(model_path, model_type, device) {
  load_detector(model_path, model_type = model_type, device = device)
}


#' Load a class list and build a category_map for detect()
#'
#' @param class_list_path string, path to a class list CSV file
#'
#' @return a Python dict (via reticulate) suitable for detect()'s
#'   category_map argument
build_category_map <- function(class_list_path) {
  class_list <- load_class_list(class_list_path)

  id_col    <- intersect(c("id", "ID", "category", "class_id"), names(class_list))[1]
  label_col <- intersect(c("class", "Class", "name", "label"), names(class_list))[1]

  if (is.na(id_col) || is.na(label_col)) {
    warning("Could not identify ID/label columns in class list -- category_map will be empty.")
    return(reticulate::dict())
  }

  stats::setNames(
    as.list(as.character(class_list[[label_col]])),
    as.character(as.integer(class_list[[id_col]]))
  )
}


#' Default category map for genuine MegaDetector models
#'
#' @return named list, id -> label, matching MegaDetector's standard
#'   empty/animal/person/vehicle categories
default_md_category_map <- function() {
  list(`0` = "empty", `1` = "animal", `2` = "person", `3` = "vehicle")
}


#' Drop duplicate candidate detections for the same bounding box
#'
#' @param detections data frame, output of detect_animals()
#' @param bbox_tolerance numeric, decimal places to round bbox coords to
#'
#' @return detections with only the top-confidence row per distinct box
drop_duplicate_boxes <- function(detections, bbox_tolerance = 3) {
  box_key <- paste(
    detections$filepath,
    round(detections$bbox_x, bbox_tolerance),
    round(detections$bbox_y, bbox_tolerance),
    round(detections$bbox_w, bbox_tolerance),
    round(detections$bbox_h, bbox_tolerance)
  )

  detections <- detections[order(-detections$conf), ]
  detections[!duplicated(box_key[order(-detections$conf)]), ]
}


#' Run the detector on a file manifest and parse the results
#'
#' @param detector an animl detector object, from load_md_detector()
#' @param files data frame, the manifest/frames to run detection on
#' @param device string, compute device to run inference on
#' @param category_map dict, id -> label. Never pass NULL/empty list().
#' @param confidence_threshold numeric, detections below this are dropped
#' @param resize_width integer, width the detector resizes images to
#' @param resize_height integer, height the detector resizes images to
#' @param batch_size integer, number of images processed per batch
#'
#' @return data frame of parsed detections merged with the input manifest
detect_animals <- function(detector, files, device,
                            category_map = reticulate::dict(),
                            confidence_threshold = 0.1,
                            resize_width = 2048, resize_height = 1440,
                            batch_size = 1) {
  mdraw <- detect(
    detector, files,
    resize_width          = resize_width,
    resize_height          = resize_height,
    batch_size             = batch_size,
    device                 = device,
    category_map           = category_map,
    confidence_threshold   = confidence_threshold
  )

  parse_detections(mdraw$detections, manifest = files)
}


#' Get a still-image file for previewing a manifest row's frame
#'
#' For a plain photo row, filepath already points to a real image file
#' -- returned unchanged. For a row that came from a VIDEO, filepath
#' still points at the original video, and extracted still images are
#' NOT reliably written to disk in every installation (confirmed by
#' direct testing). Instead, this calls a Python-side helper
#' (_animl_shiny_save_video_frame(), defined once at the top of this
#' file via reticulate::py_run_string()) that wraps animl-py's
#' get_frame_as_image() -- reads the exact frame from the video via
#' OpenCV and writes it straight to a JPEG, entirely inside Python.
#'
#' @param filepath string, the manifest row's filepath column
#' @param frame numeric/integer, the manifest row's frame column
#'
#' @return string, path to a real image file to preview, or NA if the
#'   frame couldn't be read at all
#'
#' @details The whole read -> convert -> write operation happens
#'   inside a single Python call (see the py_run_string() block near
#'   the top of this file) rather than doing it in separate R-side
#'   reticulate calls. Confirmed by direct testing: letting the numpy
#'   array cross into R and back to Python (e.g. calling cv2$cvtColor()
#'   on an R-side reference) silently widens its dtype from uint8 to
#'   int32 (R has no native 8-bit integer type), which OpenCV then
#'   rejects with "Unsupported depth of input image... depth is 4
#'   (CV_32S)" -- even after explicitly casting back with
#'   numpy$asarray(dtype=uint8). Keeping the array Python-side the
#'   whole time avoids the conversion entirely.
get_video_frame_path <- function(filepath, frame) {
  video_extensions <- c("mp4", "avi", "mov", "wmv", "mkv", "m4v", "mpg", "mpeg")
  ext <- tolower(tools::file_ext(filepath))

  if (!(ext %in% video_extensions)) {
    return(filepath)
  }

  frame_num <- if (is.na(frame)) 0L else as.integer(frame)
  out_path  <- tempfile(fileext = ".jpg")

  result <- reticulate::py$`_animl_shiny_save_video_frame`(filepath, frame_num, out_path)

  if (is.null(result)) NA_character_ else out_path
}


#' Draw a bounding box onto an image for preview
#'
#' @param filepath string, path to the source image (already resolved
#'   via get_video_frame_path() by the caller)
#' @param bbox_x,bbox_y,bbox_w,bbox_h numeric, normalized (0-1) box
#'   coordinates. Any NA skips drawing and just returns the plain image.
#'
#' @return string, path to a temp PNG file with the box drawn
draw_bbox_preview <- function(filepath, bbox_x, bbox_y, bbox_w, bbox_h) {
  img <- image_read(filepath)
  info <- image_info(img)

  has_box <- !any(is.na(c(bbox_x, bbox_y, bbox_w, bbox_h)))

  if (has_box) {
    x0 <- bbox_x * info$width
    y0 <- bbox_y * info$height
    x1 <- (bbox_x + bbox_w) * info$width
    y1 <- (bbox_y + bbox_h) * info$height

    img <- image_draw(img)
    rect(x0, y0, x1, y1, border = "red", lwd = max(2, info$width / 400))
    dev.off()
  }

  out_path <- tempfile(fileext = ".png")
  image_write(img, out_path)
  out_path
}


# ============================================================================
# UI -- defines what the user SEES. No logic runs here, just layout.
# ============================================================================
ui <- fluidPage(

  titlePanel("AniML Camera Trap Manifest Builder"),

  sidebarLayout(

    sidebarPanel(

      h4("Step 1: Build File Manifest"),

      shinyDirButton(
        id = "dir",
        label = "Select Image Folder",
        title = "Choose a folder containing camera trap images/videos"
      ),

      br(), br(),

      verbatimTextOutput("dirpath"),

      br(),

      actionButton("run", "Build File Manifest", class = "btn-primary"),

      hr(),

      helpText(
        "Selects a directory, then calls animl::build_file_manifest() ",
        "on it (with exif = TRUE) and displays the resulting manifest."
      ),

      downloadButton("download_manifest", "Download Manifest (CSV)"),

      hr(),

      h4("Step 2: Extract Frames"),

      numericInput(
        inputId = "frames_per_video",
        label   = "Frames to pull per video",
        value   = 3,
        min     = 1,
        max     = 20,
        step    = 1
      ),

      actionButton("extract", "Extract Frames", class = "btn-primary"),

      helpText(
        "Runs animl::extract_frames() on the manifest above, pulling the ",
        "chosen number of still frames from each video for classification. ",
        "Requires Step 1 (Build File Manifest) to have run first."
      ),

      hr(),

      h4("Step 3: Detect Animals"),

      shinyFilesButton(
        id     = "model_file",
        label  = "Select Detector Model",
        title  = "Choose a detector model file (.pt)",
        multiple = FALSE
      ),

      br(), br(),

      verbatimTextOutput("modelpath"),

      br(),

      shinyFilesButton(
        id       = "class_list_file",
        label    = "Select Class List (required for custom models, optional for MegaDetector)",
        title    = "Choose a class list CSV file (maps category IDs to labels)",
        multiple = FALSE
      ),

      br(), br(),

      verbatimTextOutput("classlistpath"),

      br(),

      selectInput(
        inputId  = "model_type",
        label    = "Model type",
        choices  = c(
          "Custom YOLO (v6+)"       = "yolo",
          "Custom YOLOv5"           = "yolov5",
          "MegaDetector v5"         = "mdv5",
          "MegaDetector v6"         = "mdv6"
        ),
        selected = "yolo"
      ),

      helpText(
        "Use \"Custom YOLO\" for a custom-trained model like ours -- ",
        "\"MegaDetector v5/v6\" are ONLY for genuine MegaDetector ",
        "releases. See the MODEL_TYPE note at the top of app.R."
      ),

      selectInput(
        inputId  = "device",
        label    = "Device",
        choices  = c("GPU (cuda:0)" = "cuda:0", "CPU" = "cpu"),
        selected = "cuda:0"
      ),

      numericInput(
        inputId = "confidence_threshold",
        label   = "Confidence threshold (detections below this are dropped)",
        value   = 0.1,
        min     = 0,
        max     = 1,
        step    = 0.05
      ),

      actionButton("detect", "Detect Animals", class = "btn-primary"),

      helpText(
        "Loads the selected detector model and runs it on the results ",
        "above, adding bounding boxes, confidence scores, and category ",
        "labels. Requires Step 1 (and Step 2, if your data has videos) ",
        "to have run first."
      ),

      hr(),

      h4("Step 4: View Selected Image"),

      helpText(
        "Click any row in the results table to view that image. Works ",
        "at any point in the pipeline: right after Step 1/2 shows the ",
        "plain image, and after Step 3 (Detect Animals) shows the ",
        "bounding box drawn on it too. For video-sourced rows, the ",
        "exact frame is read directly from the video."
      )
    ),

    mainPanel(
      textOutput("status"),

      DTOutput("results_table"),

      hr(),

      imageOutput("bbox_preview", height = "auto")
    )
  )
)


# ============================================================================
# SERVER -- defines what the app DOES. Runs once per user session.
# ============================================================================
server <- function(input, output, session) {

  volumes <- c(
    Home = fs::path_home(),
    "R Installation" = R.home(),
    shinyFiles::getVolumes()()
  )

  shinyDirChoose(input, "dir", roots = volumes, session = session)

  selected_dir <- reactive({
    req(input$dir)
    parseDirPath(volumes, input$dir)
  })

  output$dirpath <- renderPrint({
    if (is.integer(input$dir)) {
      cat("No folder selected yet")
    } else {
      cat("Selected:", selected_dir())
    }
  })

  manifest <- eventReactive(input$run, {
    req(selected_dir())

    withProgress(message = "Building file manifest...", value = 0.3, {
      files <- build_manifest_from_dir(selected_dir())
      incProgress(0.7)
      files
    })
  })

  results_data <- reactiveVal(NULL)
  detection_input <- reactiveVal(NULL)

  observeEvent(input$run, {
    results_data(manifest())
    detection_input(manifest())
  })

  observeEvent(input$extract, {
    req(manifest())

    withProgress(message = "Extracting frames...", value = 0.2, {
      allframes <- extract_video_frames(manifest(), input$frames_per_video)
      incProgress(0.8)
      results_data(allframes)
      detection_input(allframes)
    })
  })

  shinyFileChoose(input, "model_file", roots = volumes, session = session,
                   filetypes = c("pt"))

  selected_model <- reactive({
    req(input$model_file)
    parseFilePaths(volumes, input$model_file)$datapath
  })

  output$modelpath <- renderPrint({
    if (is.null(input$model_file) || is.integer(input$model_file)) {
      cat("No model selected yet")
    } else {
      cat("Selected:", selected_model())
    }
  })

  shinyFileChoose(input, "class_list_file", roots = volumes, session = session,
                   filetypes = c("csv"))

  selected_class_list <- reactive({
    req(input$class_list_file)
    parseFilePaths(volumes, input$class_list_file)$datapath
  })

  output$classlistpath <- renderPrint({
    if (is.null(input$class_list_file) || is.integer(input$class_list_file)) {
      cat("No class list selected (category labels will show animl's default)")
    } else {
      cat("Selected:", selected_class_list())
    }
  })

  observeEvent(input$detect, {
    req(selected_model())
    req(detection_input())

    tryCatch({
      withProgress(message = "Running detector...", value = 0.1, {
        detector <- load_md_detector(selected_model(), input$model_type, input$device)
        incProgress(0.2)

        category_map <- reticulate::dict()
        if (!is.null(input$class_list_file) && !is.integer(input$class_list_file)) {
          category_map <- build_category_map(selected_class_list())
        } else if (input$model_type %in% c("mdv5", "mdv6", "mdv1000-cedar", "mdv1000-larch",
                                            "mdv1000-sorrel", "mdv1000-redwood", "mdv1000-spruce")) {
          category_map <- default_md_category_map()
        }
        incProgress(0.1)

        if (input$model_type %in% c("mdv5", "yolov5")) {
          resize_w <- 1280
          resize_h <- 1280
        } else {
          resize_w <- 2048
          resize_h <- 1440
        }

        detections <- detect_animals(
          detector, detection_input(), input$device,
          category_map            = category_map,
          confidence_threshold     = input$confidence_threshold,
          resize_width             = resize_w,
          resize_height            = resize_h
        )

        detections <- drop_duplicate_boxes(detections)
        incProgress(0.6)

        results_data(detections)
      })
    }, error = function(e) {
      showNotification(
        paste0(
          "Detection failed -- this usually means the selected Model ",
          "type doesn't match this model file's actual architecture ",
          "(e.g. picking MegaDetector v5/v6 for a custom-trained model, ",
          "or vice versa). Try a different Model type.\n\nRaw error: ",
          conditionMessage(e)
        ),
        type = "error",
        duration = NULL
      )
    })
  })

  output$status <- renderText({
    req(results_data())
    n <- nrow(results_data())
    if (n == 0) {
      "No rows to show yet."
    } else {
      paste0("Showing ", n, " row(s). Click a row to preview its bounding box.")
    }
  })

  output$results_table <- renderDT({
    req(results_data())
    datatable(
      results_data(),
      selection = "single",
      options = list(scrollX = TRUE, pageLength = 15)
    )
  })

  selected_row <- reactive({
    req(input$results_table_rows_selected)
    results_data()[input$results_table_rows_selected, ]
  })

  output$bbox_preview <- renderImage({
    row <- selected_row()

    frame_val <- if ("frame" %in% names(row)) row$frame else NA
    real_path <- get_video_frame_path(row$filepath, frame_val)

    if (is.na(real_path)) {
      placeholder <- image_blank(width = 800, height = 450, color = "gray20")
      placeholder <- image_annotate(
        placeholder,
        paste0(
          "Could not read this frame from the video.\n\n",
          basename(row$filepath)
        ),
        gravity = "center", color = "white", size = 22
      )
      out_path <- tempfile(fileext = ".png")
      image_write(placeholder, out_path)
      return(list(src = out_path, contentType = "image/png", width = "100%"))
    }

    has_bbox_cols <- all(c("bbox_x", "bbox_y", "bbox_w", "bbox_h") %in% names(row))

    img_path <- if (has_bbox_cols) {
      draw_bbox_preview(
        real_path, row$bbox_x, row$bbox_y, row$bbox_w, row$bbox_h
      )
    } else {
      draw_bbox_preview(real_path, NA, NA, NA, NA)
    }

    list(src = img_path, contentType = "image/png", width = "100%")
  }, deleteFile = TRUE)

  output$download_manifest <- downloadHandler(
    filename = function() "file_manifest.csv",
    content = function(file) {
      req(results_data())
      write.csv(results_data(), file, row.names = FALSE)
    }
  )
}

# ============================================================================
# Launch the app by combining the ui and server pieces defined above.
# ============================================================================
shinyApp(ui, server)
