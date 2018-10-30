#define _GNU_SOURCE
#include <assert.h>
#include <elf.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <linux/limits.h>
#include <sys/mman.h>
#include <sys/types.h>
#include <sys/stat.h>

#include "trampoline_setup.h"
#include "common.h"
#include "utils.h"

static int getSymbolTable(const char* , Elf64_Shdr* , char** );
static int readElfSection(int , int , const Elf64_Ehdr* ,
                          Elf64_Shdr* , char **);

// Returns offset of symbol, or -1 on failure.
off_t
get_symbol_offset(const char *libname, const char *symbol)
{
  int rc;
  Elf64_Shdr symtab;
  Elf64_Sym symtab_entry;

  off_t result = -1;
  char *strtab = NULL;

  int fd = getSymbolTable(libname, &symtab, &strtab);
  if (fd < 0) {
    DLOG(ERROR, "Failed to file debug symbol file for %s\n", libname);
    return -1;
  }

  // Move to beginning of symbol table
  lseek(fd, symtab.sh_offset, SEEK_SET);
  for ( ; lseek(fd, 0, SEEK_CUR) - symtab.sh_offset < symtab.sh_size; ) {
    rc = read(fd, &symtab_entry, sizeof symtab_entry);
    assert(rc == sizeof(symtab_entry));
    if (strcmp(strtab + symtab_entry.st_name, symbol) == 0) {
      // found address as offset from base address
      result = symtab_entry.st_value;
      break;
    }
  }
  if (strtab) {
    free(strtab);
  }
  close(fd);
  if (result == -1) {
    DLOG(ERROR, "Failed to find symbol (%s) in %s\n", symbol, libname);
  }
  return result;
}

// Returns 0 on success, -1 on failure
int
insertTrampoline(void *from_addr, void *to_addr)
{
  int rc;

  if (!from_addr || !to_addr)
    return -1;

#if defined(__x86_64__)
  unsigned char asm_jump[] = {
    // mov    $0x1234567812345678,%rax
    0x48, 0xb8, 0x78, 0x56, 0x34, 0x12, 0x78, 0x56, 0x34, 0x12,
    // jmpq   *%rax
    0xff, 0xe0
  };
  // Beginning of address in asm_jump:
  const int addr_offset = 2;
#elif defined(__i386__)
    static unsigned char asm_jump[] = {
      0xb8, 0x78, 0x56, 0x34, 0x12, // mov    $0x12345678,%eax
      0xff, 0xe0                    // jmp    *%eax
  };
  // Beginning of address in asm_jump:
  const int addr_offset = 1;
#else
# error "Architecture not supported"
#endif

  void *page_base = (void *)ROUND_DOWN(from_addr);
  size_t page_length = PAGE_SIZE;
  if ((VA)from_addr + sizeof(asm_jump) - (VA)page_base > PAGE_SIZE) {
    // The patching instructions cross page boundary. View page as double size.
    page_length = 2 * PAGE_SIZE;
  }

  // Temporarily add write permissions
  rc = mprotect(page_base, page_length, PROT_READ | PROT_WRITE | PROT_EXEC);
  if (rc < 0) {
    DLOG(ERROR, "mprotect failed for %p at %zu. Error: %s\n",
         page_base, page_length, strerror(errno));
    return -1;
  }

  // Now, do the patching
  memcpy(from_addr, asm_jump, sizeof(asm_jump));
  memcpy((VA)from_addr + addr_offset, (void*)&to_addr, sizeof(&to_addr));

  // Finally, remove the write permissions
  rc = mprotect(page_base, page_length, PROT_READ | PROT_EXEC);
  if (rc < 0) {
    DLOG(ERROR, "mprotect failed: %s\n", strerror(errno));
    return -1;
  }
  return rc;
}

// On success, returns fd of debug file, pointers to symtab and strtab
// On failures, returns -1
static int
getSymbolTable(const char *libname, Elf64_Shdr *symtab, char **strtab)
{
  int rc;
  int fd = -1;
  int retries = 0;
  int symtab_found = 0;
  int foundDebugLib = 0;
  char debugLibName[PATH_MAX] = {0};

  char *shsectData = NULL;
  char *lname = (char*)libname;

  Elf64_Shdr sect_hdr;

  while (retries < 2) {
    fd = open(lname, O_RDONLY);

    // Reset fd to beginning and parse file header
    lseek(fd, 0, SEEK_SET);
    Elf64_Ehdr elf_hdr;
    rc = read(fd, &elf_hdr, sizeof(elf_hdr));
    assert(rc == sizeof(elf_hdr));

    // Get start of symbol table and string table
    Elf64_Off shoff = elf_hdr.e_shoff;

    // First, read the data from the shstrtab section
    // This section contains the strings corresponding to the section names
    rc = readElfSection(fd, elf_hdr.e_shstrndx,
                           &elf_hdr, &sect_hdr, &shsectData);

    lseek(fd, shoff, SEEK_SET);
    for (int i = 0; i < elf_hdr.e_shnum; i++) {
      rc = read(fd, &sect_hdr, sizeof sect_hdr);
      assert(rc == sizeof(sect_hdr));
      if (sect_hdr.sh_type == SHT_SYMTAB) {
        *symtab = sect_hdr;
        symtab_found = 1;
      } else if (sect_hdr.sh_type == SHT_STRTAB &&
                 !strcmp(&shsectData[sect_hdr.sh_name], ELF_STRTAB_SECT)) {
        // Note that there are generally three STRTAB sections in ELF binaries:
        //  1. .dynstr
        //  2. .shstrtab
        //  3. .strtab
        // We only care about the strtab section.
        Elf64_Shdr tmp;
        rc = readElfSection(fd, i, &elf_hdr, &tmp, strtab);
      } else if (sect_hdr.sh_type == SHT_PROGBITS &&
                 !strcmp(&shsectData[sect_hdr.sh_name], ELF_DEBUGLINK_SECT)) {
        // If it's the ".gnu_debuglink" section, we read it to figure out
        // the path to the debug symbol file
        Elf64_Shdr tmp;
        char *debugName = NULL;
        rc = readElfSection(fd, i, &elf_hdr, &tmp, &debugName);
        assert(debugName);
        snprintf(debugLibName, sizeof debugLibName, "%s/%s",
                 DEBUG_FILES_PATH, debugName);
        free(debugName);
        foundDebugLib = 1;
      }
    }

    if (symtab_found || !foundDebugLib) {
      break;
    }

    // Let's try again with debug library
    lname = debugLibName;
    DLOG(INFO, "Failed to find symbol table in %s. Retrying with %s...\n",
         libname, lname);
    retries++;
  }
  free(shsectData);
  if (retries == 2 && !symtab_found) {
    DLOG(ERROR, "Failed to find symbol table in %s\n", libname);
    close(fd);
    return -1;
  }
  return fd;
}

static int
readElfSection(int fd, int sidx, const Elf64_Ehdr *ehdr,
               Elf64_Shdr *shdr, char **data)
{
  off_t currOff = lseek(fd, 0, SEEK_CUR);
  off_t sidx_off = ehdr->e_shentsize * sidx + ehdr->e_shoff;
  lseek(fd, sidx_off, SEEK_SET);
  int rc = read(fd, shdr, sizeof *shdr);
  assert(rc == sizeof *shdr);
  rc = lseek(fd, shdr->sh_offset, SEEK_SET);
  if (rc > 0) {
    *data = malloc(shdr->sh_size);
    rc = lseek(fd, shdr->sh_offset, SEEK_SET);
    rc = readAll(fd, *data, shdr->sh_size);
  }
  lseek(fd, currOff, SEEK_SET);
  return *data != NULL ? 0 : -1;
}
