#ifndef ACTIVE_MODEL_H
#define ACTIVE_MODEL_H

#include "config.h"

/*
 * Conditional Model Selector
 * Selects between the pre-trained base model and the surface-calibrated fine-tuned model
 * based on the USE_FINETUNED_MODEL macro defined in config.h.
 */
#ifdef USE_FINETUNED_MODEL
  #include "model_finetuned.h"
#else
  #include "model.h"
#endif

#endif // ACTIVE_MODEL_H
