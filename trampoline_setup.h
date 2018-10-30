#ifndef TRAMPOLINE_SETUP_H
#define TRAMPOLINE_SETUP_H

#define MMAP_SYMBOL_NAME     "mmap"
#define SBRK_SYMBOL_NAME     "sbrk"
#define ELF_STRTAB_SECT      ".strtab"
#define ELF_DEBUGLINK_SECT   ".gnu_debuglink"

// FIXME: Find this path at runtime?
#define DEBUG_FILES_PATH   "/usr/lib/debug/lib/x86_64-linux-gnu"

int insertTrampoline(void* , void* );
off_t get_symbol_offset(const char* , const char* );

#endif // ifndef TRAMPOLINE_SETUP_H
