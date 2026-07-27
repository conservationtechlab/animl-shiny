# ============================================================================
# animl-shiny
# ============================================================================
#
# Pipeline UI, built one step at a time:
#   Step 1: select a folder of camera trap images/videos, then run
#           animl::build_file_manifest() on it and preview the result.
#   Step 2: pick how many frames to pull per video, then run
#           animl::extract_frames() against that manifest.
<<<<<<< HEAD
#
# The manifest (a table of every image/video file found, plus EXIF metadata
# like timestamps) from Step 1 is the input Step 2 needs. Later pipeline
# steps -- MegaDetector, species classification -- will build on Step 2's
# output the same way.
=======
#   Step 3: select a MegaDetector model file, then run animl::detect()
#           (via load_detector() + parse_detections()) against the
#           results so far, adding bounding boxes and confidence scores.
#
# The manifest (a table of every image/video file found, plus EXIF metadata
# like timestamps) from Step 1 is the input Step 2 needs; Step 2's output
# (or Step 1's, if there were no videos) feeds Step 3. Later pipeline
# steps -- species classification -- will build on Step 3's output the
# same way.
>>>>>>> 77ef911 (Add detector step (MegaDetector) to shiny app (#3))
#
# NOTE: saving intermediate outputs to disk (out_file / out_dir args,
# WorkingDirectory()'s save-file locations) is intentionally left out of
# these helpers for now -- that's a separate concern to design properly
# later, rather than something to bolt on ad hoc here.
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
#'   to be fed into MegaDetector in a later pipeline step
extract_video_frames <- function(files, frames_per_video) {
  extract_frames(
    files,
    frames      = frames_per_video,
    parallel    = TRUE,
    num_workers = parallel::detectCores()
  )
}


#' Load a MegaDetector model
#'
#' Kept separate from detect_animals() for the same reason
#' set_working_directory() is separate from build_manifest_from_dir() --
#' loading the model is a distinct setup step, not part of running
#' detection itself.
#'
#' @param model_path string, path to a MegaDetector model file (.pt)
#' @param model_type string, animl model type identifier (e.g. "mdv5", "mdv6")
#' @param device string, compute device to load the model onto
#'   (e.g. "cuda:0" for GPU, "cpu" otherwise)
#'
#' @return an animl detector object, ready to be passed to detect_animals()
load_md_detector <- function(model_path, model_type, device) {
  load_detector(model_path, model_type = model_type, device = device)
}


#' Run MegaDetector on a file manifest and parse the results
#'
#' Thin wrapper around animl's detect() + parse_detections(). Resize
#' dimensions and batch size default to the values animl's own README
#' example uses for MDv5 (1280x960, batch_size = 4).
#'
#' @param detector an animl detector object, from load_md_detector()
#' @param files data frame, the manifest/frames to run detection on
#' @param device string, compute device to run inference on
#' @param resize_width integer, width MegaDetector resizes images to
#' @param resize_height integer, height MegaDetector resizes images to
#' @param batch_size integer, number of images processed per batch
#'
#' @return data frame of parsed detections (bounding boxes + confidence),
#'   merged with the input manifest -- ready for classification in a
#'   later pipeline step
detect_animals <- function(detector, files, device,
                           resize_width = 1280, resize_height = 960,
                           batch_size = 4) {
  mdraw <- detect(
    detector, files,
    resize_width  = resize_width,
    resize_height = resize_height,
    batch_size    = batch_size,
    device        = device
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
<<<<<<< HEAD

      # ---- Step 1: folder selection + manifest controls -----------------
      h4("Step 1: Build File Manifest"),

=======
      
      # ---- Step 1: folder selection + manifest controls -----------------
      h4("Step 1: Build File Manifest"),
      
>>>>>>> 77ef911 (Add detector step (MegaDetector) to shiny app (#3))
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
<<<<<<< HEAD

      hr(),

      # ---- Step 2: frame extraction controls -------------------------
      h4("Step 2: Extract Frames"),

=======
      
      hr(),
      
      # ---- Step 2: frame extraction controls -------------------------
      h4("Step 2: Extract Frames"),
      
>>>>>>> 77ef911 (Add detector step (MegaDetector) to shiny app (#3))
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
<<<<<<< HEAD

      actionButton("extract", "Extract Frames", class = "btn-primary"),

=======
      
      actionButton("extract", "Extract Frames", class = "btn-primary"),
      
>>>>>>> 77ef911 (Add detector step (MegaDetector) to shiny app (#3))
      helpText(
        "Runs animl::extract_frames() on the manifest above, pulling the ",
        "chosen number of still frames from each video for classification. ",
        "Requires Step 1 (Build File Manifest) to have run first."
<<<<<<< HEAD
=======
      ),
      
      hr(),
      
      # ---- Step 3: detector controls ----------------------------------
      h4("Step 3: Detect Animals"),
      
      # shinyFilesButton() is shinyFiles' file-picker counterpart to
      # shinyDirButton() -- same browsing mechanism, but for picking a
      # single file (here, a MegaDetector .pt model file) instead of a
      # folder.
      shinyFilesButton(
        id     = "model_file",
        label  = "Select Detector Model",
        title  = "Choose a MegaDetector model file (.pt)",
        multiple = FALSE
      ),
      
      br(), br(),
      
      verbatimTextOutput("modelpath"),
      
      br(),
      
      selectInput(
        inputId  = "model_type",
        label    = "Model type",
        choices  = c("MegaDetector v5" = "mdv5", "MegaDetector v6" = "mdv6"),
        selected = "mdv5"
      ),
      
      selectInput(
        inputId  = "device",
        label    = "Device",
        choices  = c("GPU (cuda:0)" = "cuda:0", "CPU" = "cpu"),
        selected = "cuda:0"
      ),
      
      actionButton("detect", "Detect Animals", class = "btn-primary"),
      
      helpText(
        "Loads the selected MegaDetector model and runs it on the ",
        "results above, adding bounding boxes and confidence scores. ",
        "Requires Step 1 (and Step 2, if your data has videos) to have ",
        "run first."
>>>>>>> 77ef911 (Add detector step (MegaDetector) to shiny app (#3))
      )
    ),
    
    mainPanel(
      # A single table that reflects whichever step has run most
<<<<<<< HEAD
      # recently: the file manifest after Step 1, then updated in place
      # to show extracted frames after Step 2 -- rather than stacking a
      # second table below it.
=======
      # recently: the file manifest after Step 1, updated in place after
      # Step 2 (extracted frames) and again after Step 3 (detections) --
      # rather than stacking a separate table per step.
>>>>>>> 77ef911 (Add detector step (MegaDetector) to shiny app (#3))
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
<<<<<<< HEAD

=======
  
>>>>>>> 77ef911 (Add detector step (MegaDetector) to shiny app (#3))
  # ---- Reactive: the single table shown in the UI --------------------------
  # results_data() holds whatever should currently be displayed:
  #   - the manifest, once Step 1 has run
  #   - then the extracted frames, once Step 2 has also run
  # reactiveVal() is a plain mutable reactive value (unlike reactive()/
  # eventReactive(), which derive their value from a formula) -- we update
  # it explicitly with observeEvent() below whenever a step completes.
  results_data <- reactiveVal(NULL)
<<<<<<< HEAD

  observeEvent(input$run, {
    results_data(manifest())
  })

  observeEvent(input$extract, {
    req(manifest())  # Step 2 requires Step 1 to have already run

    withProgress(message = "Extracting frames...", value = 0.2, {
      allframes <- extract_video_frames(manifest(), input$frames_per_video)
      incProgress(0.8)
      results_data(allframes)  # replaces the manifest in the same table
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
=======
  
  observeEvent(input$run, {
    results_data(manifest())
>>>>>>> 77ef911 (Add detector step (MegaDetector) to shiny app (#3))
  })
  
  observeEvent(input$extract, {
    req(manifest())  # Step 2 requires Step 1 to have already run
    
    withProgress(message = "Extracting frames...", value = 0.2, {
      allframes <- extract_video_frames(manifest(), input$frames_per_video)
      incProgress(0.8)
      results_data(allframes)  # replaces the manifest in the same table
    })
  })
  
  # ---- Model file picker setup ---------------------------------------------
  # Same pattern as the folder picker, but shinyFileChoose() for a single
  # file instead of shinyDirChoose() for a directory. filetypes restricts
  # the browser dialog to .pt files (the format MegaDetector models ship as).
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
  
  # ---- Step 3: Detect Animals -----------------------------------------------
  # Same pattern as Steps 1 and 2: only runs on its own button click, and
  # req(results_data()) blocks it from running before earlier steps have.
  observeEvent(input$detect, {
    req(selected_model())
    req(results_data())
    
    withProgress(message = "Running MegaDetector...", value = 0.1, {
      # NOTE: this loads the model fresh on every click. Fine for now
      # while we're building the pipeline step by step, but worth
      # caching the loaded detector later if this becomes a bottleneck
      # (e.g. reusing it across multiple detection runs in one session).
      detector <- load_md_detector(selected_model(), input$model_type, input$device)
      incProgress(0.3)
      
      detections <- detect_animals(detector, results_data(), input$device)
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