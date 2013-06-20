#include "stdafx.h"
#include <windows.h>
#include <iostream>
#include "../wrapper/cuda-wrapper.h"
#include "version.h"

void Help()
{
    std::cerr << "\
USAGE: waste [OPTION] ... FILE\n\
Emulation of CUDA program FILE.\n\
\n\
Options:\n\
  -e                                Set emulator mode.\n\
  -ne                               Set non-emulator mode.\n\
  -d device                         Set the name of the device to emulate.\n\
  -t=NUMBER                         Trace CUDA memory API calls, emulator, etc, at a given level of noisiness.\n\
  -s=NUMBER, --padding-size=NUMBER  Set size of padding for buffer allocations.\n\
  -b=CHAR, --padding-byte=CHAR      Set byte of padding.\n\
  -q, --quit-on-error               Quit program on error detection.\n\
  -k, --skip-on-error               Skip over CUDA call when the usage is invalid.\n\
  -n, --non-standard-ptr            Allow computed pointers.\n\
  -v, --version                     Print out version\n\
  -x                                Start debugger.\n\
";
    exit(1);
}

int main(int argc, char * argv[])
{
    // Perform debugging of CUDA memory calls.
    argc--;
    argv++;

    if (argc == 0)
    {
        Help();
    }

    bool set_trace_all_calls = false;
    int trace_all_calls;
    bool set_padding_size = false;
    int padding_size;
    bool set_padding_byte = false;
    int padding_byte;
    bool set_quit_on_error = false;
    int quit_on_error;
    bool set_do_not_call_cuda_after_sanity_check_fail = false;
    int do_not_call_cuda_after_sanity_check_fail;
    bool set_device_pointer_to_first_byte_in_block = false;
    int device_pointer_to_first_byte_in_block;
    bool set_emulator_mode = true;
    bool set_device = false;
    bool do_debugger = false;
    bool set_num_threads = false;
    int num_threads = 0;
    char * device;
    int level = 0;
    int set_stack_size = 10 * 1024 * 1024;

    // Create a structure containing options for debug.
    while (argc > 0)
    {
        if (**argv == '-')
        {
            if (strncmp("-t=", *argv, 3) == 0)
            {
                set_trace_all_calls = true;
                trace_all_calls = true;
                level = atoi(3+*argv);
            }
            else if (strcmp("-e", *argv) == 0)
            {
                set_emulator_mode = true;
            }
            else if (strcmp("-ne", *argv) == 0)
            {
                set_emulator_mode = false;
            }
            else if (strncmp("-s=", *argv, 3) == 0)
            {
                set_padding_size = true;
                padding_size = atoi(3+*argv);
            }
            else if (strncmp("--padding-size=", *argv, 15) == 0)
            {
                set_padding_size = true;
                padding_size = atoi(15+*argv);
            }
            else if (strncmp("-b=", *argv, 3) == 0)
            {
                set_padding_byte = true;
                padding_byte = atoi(3+*argv);
            }
            else if (strncmp("-c=", *argv, 3) == 0)
            {
                set_stack_size = atoi(3+*argv);
            }
            else if (strncmp("--padding-byte=", *argv, 15) == 0)
            {
                set_padding_byte = true;
                padding_byte = atoi(15+*argv);
            }
            else if (strcmp("-q", *argv) == 0 || strcmp("--quit-on-error", *argv) == 0)
            {
                set_quit_on_error = true;
                quit_on_error = true;
            }
            else if (strcmp("-k", *argv) == 0 || strcmp("--skip-on-error", *argv) == 0)
            {
                set_do_not_call_cuda_after_sanity_check_fail = true;
                do_not_call_cuda_after_sanity_check_fail = true;
            }
            else if (strcmp("-n", *argv) == 0 || strcmp("--non-standard-ptr", *argv) == 0)
            {
                set_device_pointer_to_first_byte_in_block = true;
                device_pointer_to_first_byte_in_block = true;
            }
            else if (strcmp("-d", *argv) == 0 || strcmp("--device", *argv) == 0)
            {
                set_device = true;
                argc--; argv++;
                device = *argv;
            }
            else if (strncmp("-m=", *argv, 3) == 0)
            {
                set_num_threads = true;
                num_threads = atoi(3+*argv);
            }
            else if (strcmp("-x", *argv) == 0)
            {
                do_debugger = true;
            }
            else if (strcmp("-v", *argv) == 0 || strcmp("--version", *argv) == 0)
            {
                std::cerr << "Version " << WASTE_VERSION << "\n";
            }
            else
                Help();
            argc--; argv++;
        }
        else
            break;
    }

    // Create wrapper, used to collect options, and start the executable with the wrapper.

    CUDA_WRAPPER * cu = CUDA_WRAPPER::Singleton();

    // Set options.
    if (set_trace_all_calls)
        cu->SetTraceAllCalls(trace_all_calls);

    if (do_debugger)
        cu->SetStartDebugger();

    if (level > 0)
        cu->SetTrace(level);

    if (set_quit_on_error)
        cu->SetQuitOnError(quit_on_error);

    if (set_do_not_call_cuda_after_sanity_check_fail)
        cu->SetDoNotCallCudaAfterSanityCheckFail(do_not_call_cuda_after_sanity_check_fail);

    if (set_device_pointer_to_first_byte_in_block)
        cu->SetDevicePointerToFirstByteInBlock(device_pointer_to_first_byte_in_block);

    if (set_padding_byte)
        cu->SetPaddingByte(padding_byte);

    if (set_device)
        cu->RunDevice(device);

    if (set_num_threads)
        cu->SetEmulationThreads(num_threads);

    cu->SetStackSize(set_stack_size);

    cu->SetEmulationMode(set_emulator_mode);

    if (argc == 0)
    {
        std::cout << "no program specified.\n";
        return 0;
    }


    // combine rest of args into one string.
    char * command;
    int len = 0;
    for (char ** av = argv; *av != 0; ++av)
    {
        len += strlen(*av) + 3;
    }
    command = (char *)malloc(len + 1);
    for (char ** av = argv; *av != 0; ++av)
    {
        if (av == argv)
            strcpy(command, *av);
        else
        {
            strcat(command, " \"");
            strcat(command, *av);
            strcat(command, "\"");
        }
    }

    std::cout << "Staring program " << command << "\n";

    // Start process.
    HANDLE hProcess = cu->StartProcess(command); // need to fix to add program args.

    // Wait for the program we spawned to finish.
    WaitForSingleObject( hProcess, INFINITE );

    return 0;
}

