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
#   ever saw it, which is why a confirmed corvid photo showed category=4
#   ("coyote") instead of the correct category=3 ("corvid") -- the raw
#   model output was actually right, the +1 shift broke it.
#
#   Per load_detector()'s own docstring: "for yolo models v6+, use
#   'yolo', for v5, use 'yolovv5'". Custom (non-MegaDetector) models
#   should use "yolo" (v6+) or "yolov5" (v5) -- NOT "mdv5"/"mdv6", which
#   are reserved for genuine MegaDetector releases and apply the +1
#   shift unconditionally.
#
# CATEGORY LABELS -- also found in the same source read: detect() has a
# native `category_map` parameter (a dict of {category_id: label}) that
# gets applied directly during detection -- this replaces our earlier
# custom apply_class_labels()/offset guessing entirely. We build this
# map straight from the class list CSV's id/class columns and pass it
# into detect(); animl-py's own code explicitly handles string-keyed
# maps "ie from reticulate", so this works cleanly from R.
#
# CONFIDENCE FILTERING -- detect() also has a native
# `confidence_threshold` parameter that drops weak detections before
# they're even returned. When every detection in an image falls below
# that threshold, animl-py itself returns category=None/
# category_label="empty" for that image -- so passing this natively
# gives the same "show empty rather than a weak guess" behavior we were
# previously hand-rolling after the fact, without extra custom logic.
#
# Required packages:
#   install.packages(c("shiny", "shinyFiles", "DT", "fs"))
#   install.packages("animl")   # or devtools::install_github("conservationtechlab/animl")
# ============================================================================

library(shiny)
library(shinyFiles)
library(DT)
library(fs)
library(animl)


# ============================================================================
# Helper functions
# ============================================================================
# Pulled out of the reactive blocks below and documented with roxygen2 tags
# so they read the same way animl-py's docstrings do -- what goes in, what
# comes out, and any side effects worth knowing about. RStudio can scaffold
# this skeleton for you: put your cursor inside a function and use
# Code -> Insert Roxygen Skeleton.

#' Set up animl's working directory for a folder of camera trap media
#'
#' Kept separate from build_manifest_from_dir() since it's really a
#' distinct setup step, not part of building the manifest itself.
#'
#' @param imagedir string, path to a folder of camera trap images/videos
#'
#' @details WorkingDirectory() does not return an object -- it assigns
#'   global variables into whatever environment you pass it. We pass
#'   globalenv() so those variables are available afterwards. We're not
#'   relying on any of those variables yet (see note at top of file re:
#'   saving intermediate outputs) -- this just satisfies animl's setup step.
set_working_directory <- function(imagedir) {
  WorkingDirectory(imagedir, globalenv())
}


#' Build a file manifest for a directory of camera trap media
#'
#' Thin wrapper around animl's build_file_manifest().
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
#' Thin wrapper around animl's extract_frames().
#'
#' @param files data frame, the manifest produced by build_manifest_from_dir()
#' @param frames_per_video integer, number of frames to sample per video
#'
#' @return data frame of still frames (one row per extracted frame), ready
#'   to be fed into the detector in a later pipeline step
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
#' Kept separate from detect_animals() for the same reason
#' set_working_directory() is separate from build_manifest_from_dir() --
#' loading the model is a distinct setup step, not part of running
#' detection itself.
#'
#' @param model_path string, path to a detector model file (.pt)
#' @param model_type string, animl model type identifier. Use "mdv5"/
#'   "mdv6" ONLY for genuine MegaDetector releases (these apply an
#'   automatic +1 category shift animl-py assumes for MD's own
#'   empty-at-0 convention). For a custom-trained model like ours, use
#'   "yolo" (YOLO v6+) or "yolov5" (YOLO v5) instead -- see the
#'   MODEL_TYPE note at the top of this file.
#' @param device string, compute device to load the model onto
#'   (e.g. "cuda:0" for GPU, "cpu" otherwise)
#'
#' @return an animl detector object, ready to be passed to detect_animals()
load_md_detector <- function(model_path, model_type, device) {
  load_detector(model_path, model_type = model_type, device = device)
}


#' Load a class list and build a category_map for detect()
#'
#' Wraps animl's load_class_list(), then reshapes it into the named
#' list format detect()'s category_map parameter expects: id -> class
#' name. Passing this into detect() directly (rather than post-hoc
#' merging labels onto results ourselves) is animl-py's own intended
#' mechanism for custom-model category labels -- see the CATEGORY
#' LABELS note at the top of this file.
#'
#' @param class_list_path string, path to a class list CSV file. Must
#'   have an id-like column and a class/label-like column.
#'
#' @return a named list suitable for detect()'s category_map argument,
#'   e.g. list(`0` = "bird", `1` = "bobcat", ...)
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
#' MegaDetector's own standard categories, used as a fallback when
#' model_type is "mdv5"/"mdv6" (or an mdv1000 variant) and no custom
#' class list was selected -- without this, detect() crashes for MD
#' models run without a class list, since our category_map would
#' otherwise be NULL.
#'
#' @return named list, id -> label, matching MegaDetector's standard
#'   empty/animal/person/vehicle categories
default_md_category_map <- function() {
  list(`0` = "empty", `1` = "animal", `2` = "person", `3` = "vehicle")
}

#' Drop duplicate candidate detections for the same bounding box
#'
#' The model can occasionally return multiple ranked category guesses
#' for what is really the same detected region (near-identical bbox
#' coordinates, different category/conf) -- this looks like separate
#' detections in the results table but isn't. Keeps only the
#' highest-confidence guess per (filepath, rounded bbox) group. Kept as
#' a safety net even after fixing the category_map/model_type issue,
#' since it's a distinct, separately-confirmed behavior.
#'
#' @param detections data frame, output of detect_animals()
#' @param bbox_tolerance numeric, decimal places to round bbox coords to
#'   before grouping -- tiny floating-point differences between "the
#'   same" box shouldn't count as different boxes
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
  
  # Order by confidence descending, then keep the first (highest-conf)
  # row per box_key.
  detections <- detections[order(-detections$conf), ]
  detections[!duplicated(box_key[order(-detections$conf)]), ]
}


#' Run the detector on a file manifest and parse the results
#'
#' Thin wrapper around animl's detect() + parse_detections(). Passes
#' category_map and confidence_threshold straight into detect() --
#' animl-py's own native mechanisms for category labels and confidence
#' filtering -- rather than the custom post-hoc logic this app used
#' previously. See the notes at the top of this file for why.
#'
#' @param detector an animl detector object, from load_md_detector()
#' @param files data frame, the manifest/frames to run detection on
#' @param device string, compute device to run inference on
#' @param category_map named list, id -> label, from build_category_map().
#'   Pass NULL to skip (category_label will show animl's default,
#'   typically "unknown" for non-empty detections).
#' @param confidence_threshold numeric, detections below this are
#'   dropped by detect() itself before being returned; an image with no
#'   detections surviving this threshold comes back as
#'   category_label = "empty" natively.
#' @param resize_width integer, width the detector resizes images to.
#'   Defaults to 2048, the native width of these photos (exactly
#'   divisible by 32, a YOLO stride requirement).
#' @param resize_height integer, height the detector resizes images to.
#'   Defaults to 1440, the native height (also divisible by 32).
#' @param batch_size integer, number of images processed per batch
#'
#' @return data frame of parsed detections (bounding boxes + confidence
#'   + category labels), merged with the input manifest -- ready for
#'   classification in a later pipeline step
#'
#' @details detect() returns a named list with two elements --
#'   $detections (the actual per-image results) and $failed_files (any
#'   images the detector couldn't process). parse_detections() expects
#'   just the $detections list, not the wrapper -- passing the wrapper
#'   directly throws "MD results input must be list" from the Python side.
detect_animals <- function(detector, files, device,
                           category_map = reticulate::dict(),
                           confidence_threshold = 0.1,
                           resize_width = 2048, resize_height = 1440,
                           batch_size = 1) {
  mdraw <- detect(
    detector, files,
    resize_width          = resize_width,
    resize_height         = resize_height,
    batch_size             = batch_size,
    device                 = device,
    category_map           = category_map,
    confidence_threshold    = confidence_threshold
  )
  
  parse_detections(mdraw$detections, manifest = files)
}


# ============================================================================
# UI -- defines what the user SEES. No logic runs here, just layout.
# ============================================================================
#
# fluidPage() is the standard Shiny page container -- responsive width,
# Bootstrap styling out of the box. Note the capital P: fluidPage(), not
# fluidpage() -- R function names are case-sensitive.
ui <- fluidPage(
  
  titlePanel("AniML Camera Trap Manifest Builder"),
  
  # sidebarLayout() gives us the classic two-column Shiny layout:
  # a narrow sidebarPanel() for controls, and a wider mainPanel() for output.
  sidebarLayout(
    
    sidebarPanel(
      
      # ---- Step 1: folder selection + manifest controls -----------------
      h4("Step 1: Build File Manifest"),
      
      # shinyDirButton() draws a button that, when clicked, opens a folder
      # browser dialog (server-side, so it works even if this app is
      # deployed to a browser and not just run locally in RStudio).
      # id = "dir" is how we'll refer to whatever gets picked, on the
      # server side, as input$dir.
      shinyDirButton(
        id = "dir",
        label = "Select Image Folder",
        title = "Choose a folder containing camera trap images/videos"
      ),
      
      br(), br(),
      
      # verbatimTextOutput() is the UI-side placeholder for text we'll
      # generate on the server with renderPrint(). The "dirpath" id here
      # must match output$dirpath in the server function below --
      # every render*() / *Output() pair is linked by a matching id string.
      verbatimTextOutput("dirpath"),
      
      br(),
      
      # actionButton() just counts clicks -- every time it's clicked,
      # input$run increments by 1. It doesn't do anything by itself;
      # the server watches for that increment (see eventReactive below).
      actionButton("run", "Build File Manifest", class = "btn-primary"),
      
      hr(),
      
      helpText(
        "Selects a directory, then calls animl::build_file_manifest() ",
        "on it (with exif = TRUE) and displays the resulting manifest."
      ),
      
      # downloadButton() pairs with downloadHandler() on the server --
      # clicking it triggers a file save dialog in the browser.
      downloadButton("download_manifest", "Download Manifest (CSV)"),
      
      hr(),
      
      # ---- Step 2: frame extraction controls -------------------------
      h4("Step 2: Extract Frames"),
      
      # numericInput() gives the user a plain number field (with up/down
      # arrows) instead of a slider -- a good fit here since "frames per
      # video" is a small, precise integer rather than a range to explore.
      numericInput(
        inputId = "frames_per_video",
        label   = "Frames to pull per video",
        value   = 3,     # sensible default, matches the animl README example
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
      
      # ---- Step 3: detector controls ----------------------------------
      h4("Step 3: Detect Animals"),
      
      # shinyFilesButton() is shinyFiles' file-picker counterpart to
      # shinyDirButton() -- same browsing mechanism, but for picking a
      # single file (here, a detector .pt model file) instead of a
      # folder.
      shinyFilesButton(
        id     = "model_file",
        label  = "Select Detector Model",
        title  = "Choose a detector model file (.pt)",
        multiple = FALSE
      ),
      
      br(), br(),
      
      verbatimTextOutput("modelpath"),
      
      br(),
      
      # Custom detectors (species-specific models, unlike generic
      # MegaDetector) typically ship with their own class list CSV
      # mapping category IDs to real class names. Required now (rather
      # than optional) since category_map is passed natively into
      # detect() -- see the CATEGORY LABELS note at the top of this file.
      shinyFilesButton(
        id       = "class_list_file",
        label    = "Select Class List",
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
        "releases, since those apply an automatic +1 category shift ",
        "that a custom model's class list doesn't expect. See the ",
        "MODEL_TYPE note at the top of app.R for the full explanation."
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
        "labels (from the class list). Requires Step 1 (and Step 2, if ",
        "your data has videos) to have run first."
      )
    ),
    
    mainPanel(
      # A single table that reflects whichever step has run most
      # recently: the file manifest after Step 1, updated in place after
      # Step 2 (extracted frames) and again after Step 3 (detections) --
      # rather than stacking a separate table per step.
      textOutput("status"),
      DTOutput("results_table")
    )
  )
)


# ============================================================================
# SERVER -- defines what the app DOES. Runs once per user session.
# ============================================================================
#
# input   = everything the user has clicked/typed/selected in the UI
# output  = everything we render back to the UI (must match *Output() ids)
# session = info/hooks tied to this specific browser connection
server <- function(input, output, session) {
  
  # ---- Folder browsing setup ----------------------------------------------
  # shinyFiles needs to know which top-level "roots" it's allowed to browse
  # from (for security -- you don't want a web app browsing a server's
  # entire filesystem by default). We offer the user's home folder, the
  # R installation folder, and whatever drives shinyFiles auto-detects
  # (C:\, D:\, etc. on Windows).
  volumes <- c(
    Home = fs::path_home(),
    "R Installation" = R.home(),
    shinyFiles::getVolumes()()
  )
  
  # This wires up input$dir to actually respond to shinyDirButton clicks,
  # using the roots we just defined.
  shinyDirChoose(input, "dir", roots = volumes, session = session)
  
  # ---- Reactive: the currently selected folder ----------------------------
  # reactive() creates a value that automatically recalculates whenever
  # something it depends on (here, input$dir) changes. Think of it like a
  # lazy, auto-updating variable. We call it as selected_dir() elsewhere,
  # like calling a function, even though it behaves like reactive data.
  selected_dir <- reactive({
    # req() is a guard clause: if input$dir isn't set yet (user hasn't
    # picked a folder), stop here quietly instead of throwing an error.
    req(input$dir)
    
    # shinyFiles gives us back a compact internal representation of the
    # chosen path; parseDirPath() converts it into an actual usable
    # file system path string.
    parseDirPath(volumes, input$dir)
  })
  
  # ---- Output: show which folder is selected ------------------------------
  # renderPrint() captures whatever gets printed/cat()'d and sends it to
  # the matching verbatimTextOutput("dirpath") in the UI.
  output$dirpath <- renderPrint({
    if (is.integer(input$dir)) {
      # input$dir starts out as an integer placeholder before any
      # selection has been made -- this is the "nothing picked yet" state.
      cat("No folder selected yet")
    } else {
      cat("Selected:", selected_dir())
    }
  })
  
  # ---- Reactive: the file manifest, built only on button click ------------
  # eventReactive() is like reactive(), but it only recalculates when a
  # SPECIFIC trigger fires -- here, input$run (the action button) -- rather
  # than any time any dependency changes. This is what makes the manifest
  # build "on demand" instead of re-running every time selected_dir()
  # changes (e.g. while the user is still browsing folders).
  manifest <- eventReactive(input$run, {
    req(selected_dir())  # don't run if no folder has been picked
    
    # withProgress()/incProgress() show a progress bar in the UI while
    # this block runs -- purely cosmetic, doesn't affect the logic.
    withProgress(message = "Building file manifest...", value = 0.3, {
      files <- build_manifest_from_dir(selected_dir())
      incProgress(0.7)
      files  # eventReactive returns whatever the block's last line is
    })
  })
  
  # ---- Reactive: the single table shown in the UI --------------------------
  # results_data() holds whatever should currently be displayed:
  #   - the manifest, once Step 1 has run
  #   - then the extracted frames, once Step 2 has also run
  #   - then the detections, once Step 3 has also run
  # reactiveVal() is a plain mutable reactive value (unlike reactive()/
  # eventReactive(), which derive their value from a formula) -- we update
  # it explicitly with observeEvent() below whenever a step completes.
  results_data <- reactiveVal(NULL)
  
  # Separate from results_data(): this holds the clean, pre-detection
  # data (manifest or extracted frames) that Step 3 should always detect
  # against -- NOT whatever is currently displayed. Without this,
  # clicking "Detect Animals" more than once would re-run detection on
  # the previous detection output (which already has multiple rows per
  # image), multiplying rows on every click instead of replacing them.
  detection_input <- reactiveVal(NULL)
  
  observeEvent(input$run, {
    results_data(manifest())
    detection_input(manifest())
  })
  
  observeEvent(input$extract, {
    req(manifest())  # Step 2 requires Step 1 to have already run
    
    withProgress(message = "Extracting frames...", value = 0.2, {
      allframes <- extract_video_frames(manifest(), input$frames_per_video)
      incProgress(0.8)
      results_data(allframes)  # replaces the manifest in the same table
      detection_input(allframes)
    })
  })
  
  # ---- Model file picker setup ---------------------------------------------
  # Same pattern as the folder picker, but shinyFileChoose() for a single
  # file instead of shinyDirChoose() for a directory. filetypes restricts
  # the browser dialog to .pt files.
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
  
  # ---- Class list file picker setup -----------------------------------------
  # Same pattern as the model file picker.
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
  
  # ---- Step 3: Detect Animals -----------------------------------------------
  # Same pattern as Steps 1 and 2: only runs on its own button click, and
  # req(detection_input()) blocks it from running before earlier steps have.
  observeEvent(input$detect, {
    req(selected_model())
    req(detection_input())
    
    withProgress(message = "Running detector...", value = 0.1, {
      # NOTE: this loads the model fresh on every click. Fine for now
      # while we're building the pipeline step by step, but worth
      # caching the loaded detector later if this becomes a bottleneck
      # (e.g. reusing it across multiple detection runs in one session).
      detector <- load_md_detector(selected_model(), input$model_type, input$device)
      incProgress(0.2)
      
      # Build the category_map from the class list, if one was
      # selected -- passed natively into detect() rather than merged
      # on after the fact (see CATEGORY LABELS note at top of file).
      category_map <- reticulate::dict()
      if (!is.null(input$class_list_file) && !is.integer(input$class_list_file)) {
        category_map <- build_category_map(selected_class_list())
      } else if (input$model_type %in% c("mdv5", "mdv6", "mdv1000-cedar", "mdv1000-larch", "mdv1000-sorrel", "mdv1000-redwood", "mdv1000-spruce")) {
        category_map <- default_md_category_map()
      }
      incProgress(0.1)
      
      # Resize dimensions depend on model architecture -- MDv5/YOLOv5
      # requires a SQUARE input (the documented MegaDetector v5
      # standard is 1280x1280); our custom YOLO11 model runs fine at
      # native resolution (2048x1440). Mixing these up throws a
      # PyTorch tensor-shape-mismatch error from inside the model.
      if (input$model_type %in% c("mdv5", "yolov5")) {
        resize_w <- 1280
        resize_h <- 1280
      } else {
        resize_w <- 2048
        resize_h <- 1440
      }
      
      # Always detect against detection_input() (the frozen pre-detection
      # data), never results_data() -- keeps repeated clicks idempotent
      # instead of compounding on the previous detection output.
      detections <- detect_animals(
        detector, detection_input(), input$device,
        category_map          = category_map,
        confidence_threshold   = input$confidence_threshold,
        resize_width             = resize_w,
        resize_height            = resize_h
      )
      
      # The model can return multiple ranked category guesses for the
      # same physical detection (near-identical bbox, different
      # category/conf) -- collapse those down to just the top guess per
      # box before anything else, so downstream steps see one row per
      # real detected object.
      detections <- drop_duplicate_boxes(detections)
      incProgress(0.6)
      
      results_data(detections)  # replaces whatever was in the table before
    })
  })
  
  # ---- Output: summary line -------------------------------------------------
  output$status <- renderText({
    req(results_data())
    n <- nrow(results_data())
    if (n == 0) {
      "No rows to show yet."
    } else {
      paste0("Showing ", n, " row(s).")
    }
  })
  
  # ---- Output: the interactive table ---------------------------------------
  # renderDT() / DTOutput() is the DT-package equivalent of
  # renderTable()/tableOutput(), but with sorting, searching, and paging.
  output$results_table <- renderDT({
    req(results_data())
    datatable(results_data(), options = list(scrollX = TRUE, pageLength = 15))
  })
  
  # ---- Output: CSV download ---------------------------------------------
  # downloadHandler() needs two pieces:
  #   filename -> what the saved file should be called
  #   content  -> a function that writes the data to the temp `file` path
  #               Shiny hands it; Shiny then streams that file to the user
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
# This is the line RStudio's "Run App" button actually executes.
# ============================================================================
shinyApp(ui, server)