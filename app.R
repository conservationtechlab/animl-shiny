# ============================================================================
# animl-shiny
# ============================================================================
#
# Pipeline UI, built one step at a time:
#   Step 1: select a folder of camera trap images/videos, then run
#           animl::build_file_manifest() on it and preview the result.
#   Step 2: pick how many frames to pull per video, then run
#           animl::extract_frames() against that manifest.
#
# The manifest (a table of every image/video file found, plus EXIF metadata
# like timestamps) from Step 1 is the input Step 2 needs. Later pipeline
# steps -- MegaDetector, species classification -- will build on Step 2's
# output the same way.
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

#' Build a file manifest for a directory of camera trap media
#'
#' Thin wrapper around animl's WorkingDirectory() + build_file_manifest().
#' Exists mainly so the Shiny server code calls one function instead of
#' juggling WorkingDirectory()'s global-variable side effect inline.
#'
#' @param imagedir string, path to a folder of camera trap images/videos
#'
#' @return data frame with one row per file found (FilePath, EXIF fields
#'   such as DateTimeOriginal, etc.)
#'
#' @details WorkingDirectory() does not return an object -- it assigns
#'   global variables (filemanifest_file, imageframes_file, vidfdir, ...)
#'   into whatever environment you pass it. We pass globalenv() so those
#'   variables are available afterwards, including to
#'   extract_video_frames() below.
build_manifest_from_dir <- function(imagedir) {
  WorkingDirectory(imagedir, globalenv())
  
  build_file_manifest(
    imagedir,
    out_file = filemanifest_file,
    exif = TRUE
  )
}


#' Extract still frames from the videos in a file manifest
#'
#' Thin wrapper around animl's extract_frames(), using the save-file
#' locations that build_manifest_from_dir() (via WorkingDirectory())
#' already set up as global variables.
#'
#' @param files data frame, the manifest produced by build_manifest_from_dir()
#' @param frames_per_video integer, number of frames to sample per video
#'
#' @return data frame of still frames (one row per extracted frame), ready
#'   to be fed into MegaDetector in a later pipeline step
extract_video_frames <- function(files, frames_per_video, imagedir) {
  frame_out_dir <- file.path(imagedir, "extracted_frames")
  dir.create(frame_out_dir, showWarnings = FALSE)
  extract_frames(
    files,
    out_dir  = frame_out_dir,
    out_file = imageframes_file,
    frames   = frames_per_video,
    parallel = TRUE,
    num_workers  = parallel::detectCores()
  )
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
      )
    ),
    
    mainPanel(
      textOutput("status"),        # short "Found N file(s)" summary line
      DTOutput("manifest_table"),  # the interactive results table
      
      hr(),
      
      textOutput("extract_status"),  # "Extracted N frame(s)" summary line
      DTOutput("frames_table")       # frame-extraction results
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
  
  # ---- Output: summary line ("Found N file(s)") ----------------------------
  output$status <- renderText({
    req(manifest())
    paste0("Found ", nrow(manifest()), " file(s).")
  })
  
  # ---- Output: the interactive table ---------------------------------------
  # renderDT() / DTOutput() is the DT-package equivalent of
  # renderTable()/tableOutput(), but with sorting, searching, and paging.
  output$manifest_table <- renderDT({
    req(manifest())
    datatable(manifest(), options = list(scrollX = TRUE, pageLength = 15))
  })
  
  # ---- Output: CSV download ---------------------------------------------
  # downloadHandler() needs two pieces:
  #   filename -> what the saved file should be called
  #   content  -> a function that writes the data to the temp `file` path
  #               Shiny hands it; Shiny then streams that file to the user
  output$download_manifest <- downloadHandler(
    filename = function() "file_manifest.csv",
    content = function(file) {
      req(manifest())
      write.csv(manifest(), file, row.names = FALSE)
    }
  )
  
  # ==========================================================================
  # Step 2: Extract Frames
  # ==========================================================================
  # Same pattern as Step 1: an eventReactive() tied to its own button
  # (input$extract), so it only runs when the user explicitly asks it to --
  # not automatically just because manifest() or input$frames_per_video changed.
  extracted_frames <- eventReactive(input$extract, {
    # req(manifest()) does double duty here: it stops this block from
    # running if Step 1 hasn't produced a manifest yet, which is the
    # "Requires Step 1 to have run first" rule mentioned in the UI help text.
    req(manifest())
    
    withProgress(message = "Extracting frames...", value = 0.2, {
      allframes <- extract_video_frames(manifest(), input$frames_per_video, selected_dir())
      incProgress(0.8)
      allframes
    })
  })
  
  # ---- Output: frame-extraction summary line ------------------------------
  output$extract_status <- renderText({
    req(extracted_frames())
    paste0("Extracted ", nrow(extracted_frames()), " frame(s).")
  })
  
  # ---- Output: frame-extraction results table ------------------------------
  output$frames_table <- renderDT({
    req(extracted_frames())
    datatable(extracted_frames(), options = list(scrollX = TRUE, pageLength = 15))
  })
}

# ============================================================================
# Launch the app by combining the ui and server pieces defined above.
# This is the line RStudio's "Run App" button actually executes.
# ============================================================================
shinyApp(ui, server)