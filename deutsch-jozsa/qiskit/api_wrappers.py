# *****************************************************************************
#
# 2022 @ Quantinuum.
# This software and all information and expression are the property of
# Quantinuum, are Quantinuum Confidential & Proprietary,
# contain trade secrets and may not, in whole or in part, be licensed,
# used, duplicated, disclosed, or reproduced for any purpose without prior
# written permission of Quantinuum.
# All Rights Reserved.
#
# *****************************************************************************
""""
Functions used to submit jobs with the Quantinuum API.

"""
import asyncio
import json
import sys
import getpass
import jwt
import datetime
import time
from http import HTTPStatus
from typing import Optional, Any
from dotenv import load_dotenv
load_dotenv()

import os
import requests
import websockets
import nest_asyncio
import keyring
import msal

# Version of the API wrapper file
version = "1.0.1"

USERNAME = os.environ.get("QUANTINUUM_USERNAME")
PASSWORD = os.environ.get("QUANTINUUM_PASSWORD")

# This is necessary for use in Jupyter notebooks to allow for nested asyncio loops
nest_asyncio.apply()


class QuantinuumAPI:

    JOB_DONE = ['failed', 'completed', 'canceled']

    DEFAULT_API_URL = 'https://qapi.quantinuum.com/'
    DEFAULT_TIME_SAFETY = 30 # Default safety factor (in seconds) to token expiration before a refresh

    AZURE_AD_APP_ID = "4ae73294-a491-45b7-bab4-945c236ee67a"
    AZURE_AD_AUTHORITY = "https://login.microsoftonline.com/common"
    AZURE_AD_SCOPE =  ["User.Read"]
    AZURE_PROVIDER = 'microsoft'

    RESULTS_FORMAT = ["histogram-flat"] # the supported values for the results_format parameter used during job retrieval

    TC_REQUIRED = 38 # terms and conditions are required 
    PROVIDER_TOKEN_INVALID = 39 # federated provider token is invalid
    MFA_CODE_REQUIRED = 73 # mfa verification code is required during login
    MFA_CODE_INVALID = 74 # mfa verification code is invalid or expired
    
    def __init__(self,
                 user_name: str = None,
                 token: str = None,
                 machine: str = None,
                 api_url: str = None,
                 api_version: int = 1,
                 shots: int = 100,
                 use_websocket: bool = True,
                 time_safety: int = None,
                 provider: str = None):
        """Initialize and login to the Quantinuum API interface

        All arguments are optional

        Arguments:
            user_name (str): User e-mail used to register
            token (str): Token used to refresh id token
            url (str): Url of the API including version: https://qapi.quantinuum.com/v1/
            shots (int): Default number of shots for submitted experiments
            use_websockets: Whether to default to using websockets to reduce traffic
        """
        try:
            import config
        except:
            config = None

        self.api_version = self._get_val(config, key='api_version', kwarg_option=api_version)

        api_url = self._get_val(config, key='api_url', kwarg_option=api_url, default=self.DEFAULT_API_URL)
        self.url = f'{api_url}v{self.api_version}/'
        self.keyring_service = f'HQS-API:{self.url}'

        self.user_name = self._get_val(config, key='user_name', kwarg_option=user_name, default=None)
        self.refresh_token = self._get_token('refresh_token')

        self.shots = self._get_val(config, key='shots', kwarg_option=shots)
        self.machine = self._get_val(config, key='machine', kwarg_option=machine)
        self.use_websocket = self._get_val(config, key='use_websocket', kwarg_option=use_websocket)
        self.time_safety_factor = self._get_val(config, key='time_safety', kwarg_option=time_safety,
                                                default=self.DEFAULT_TIME_SAFETY)
        self.ws_timeout = self._get_val(config, key='ws_timeout', kwarg_option=None, default=180)
        self.retry_timeout = self._get_val(config, key='retry_timeout', kwarg_option=None, default=4)
        self.provider = self._get_val(config, key='provider', kwarg_option=None, default=provider)
        self.recently_submitted = []

        self.login()

    @staticmethod
    def _get_val(config: object,
                 key: str,
                 kwarg_option: Optional[Any] = None,
                 default: Optional[Any] = None) -> Any:
        """ Use `kwarg_option` if evaluates to True (i.e., `kwarg_option` is given), 
            otherwise use what is in config if it exists but, if not
            in config, use the default value.
        """

        return kwarg_option if kwarg_option else getattr(config, key, default)

    def _request_tokens(self, body):
        """ Method to send login request to machine api and save tokens. """
        try:
            # send request to login
            response = requests.post(
                f'{self.url}/login',
                json.dumps(body),
            )
            
            # reset body to delete credentials
            body = {}
                        
            if response.status_code != HTTPStatus.OK:
                return response.status_code, response.json()
            
            else:
                print("***Successfully logged in***")
                self._save_tokens(response.json()['id-token'], response.json()['refresh-token'])
                return response.status_code, None
            
        except requests.exceptions.RequestException as e:
            print(e)
            return None, None
        
    def _get_credentials(self):
        """ Method to ask for user's credentials """
        if not self.user_name:
            user_name = USERNAME
        else:
            user_name = self.user_name
        pwd = PASSWORD
        return user_name, pwd

    def _get_mfa_code(self):
        """ Method to ask for user's multi-factor authentication code """

        mfa_code = input('Enter your MFA verification code: ')
        
        return mfa_code
    
    def _authenticate(self, action=None):
        """ This method makes requests to refresh or get new id-token.
        If a token refresh fails due to token being expired, credentials
        get requested from user.
        """
        # login body
        body = {}
        user_name = ""
        pwd = ""
        provider_token = None
        if action == 'refresh':
            body['refresh-token'] = self.refresh_token
            print("Attempting to get new ID token using stored credentials.")
        else:
            # ask user for credentials before making login request
            print("Stored credentials expired or not available, requesting login credentials.")
            user_name, pwd, provider_token = "", "", ""
            if not self.provider:
                # begin native authentication flow
                user_name, pwd = self._get_credentials()
                body['email'] = user_name
                body['password'] = pwd
            else:
                # begin federated authentication flow
                user_name, provider_token = self._federated_login()
                body['provider-token'] = provider_token
            
        # send login request to API
 
        status_code, message = self._request_tokens(body)

        # clear the request body, but hold onto email/password in case of MFA is enabled so we don't prompt the user again
        body = {}
        
        if status_code != HTTPStatus.OK:
            # check if we got an error due to issues with the refresh token
            if status_code == HTTPStatus.FORBIDDEN:
                print("Stored credentials incorrect or expired, requesting login credentials.")
                print(message.get('error', {}).get('text', 'Request forbidden'))
            
                # ask user for credentials to login again
                user_name, pwd = self._get_credentials()
                body['email'] = user_name
                body['password'] = pwd
                
                # send login request to API
                status_code, message = self._request_tokens(body)

            if status_code == HTTPStatus.BAD_REQUEST:
                error = message.get('error')

                # handle an failed federated login attempt
                if error['code'] == self.PROVIDER_TOKEN_INVALID:
                    print("Unable to complete federated authentication")
                    print(message.get('error', {}).get('text', 'Request forbidden'))

                    # ask user to complete federated login again. Perhaps they used the wrong account
                    user_name, provider_token = self._federated_login()
                    body['provider-token'] = provider_token

                    # send login request to API with the provider token
                    status_code, message = self._request_tokens(body)

            if status_code == HTTPStatus.UNAUTHORIZED:

                # check if we got an error due to multi-factor authentication (MFA) being enabled 
                error = message.get('error')
                if error['code'] == self.MFA_CODE_REQUIRED:
                    # begin multi-factor authentication login flow
                    status_code, message = self._mfa_login(user_name, pwd)

                elif error['code'] == self.TC_REQUIRED:
                    # the user is not compliant with terms and conditions
                    url = message.get('error', {}).get('url', '')
                    # let them know what needs to happen in order to have a successful login next time
                    print('Terms and conditions must be accepted on UI (' + url + ') before logging in')
                    raise RuntimeError(f'Request unauthorized. Terms and conditions required', HTTPStatus.UNAUTHORIZED)

                else:
                    # The user is unauthorized to complete authentication for another reason (such as not being compliant with terms and conditions)
                    print("Unable to complete authentication.")
                    print(message.get('error', {}).get('text', 'Request unauthorized'))


        # clear credentials
        user_name = None
        pwd = None
        provider_token = None
        
        if status_code != HTTPStatus.OK:
            print("Failed to load or request valid credentials.")
            raise RuntimeError(f'HTTP error while logging in:', status_code)

    def _get_token(self, token_name:str):
        """ Method to retrieve id and refresh tokens from system's keyring service.
        Windows keyring backend has a length limitation on passwords.
        To avoid this, passwords get split up into two credentials.
        """
        
        token = None
        
        token_first = keyring.get_password(self.keyring_service, f'{token_name}_first')
        token_second = keyring.get_password(self.keyring_service, f'{token_name}_second')
        
        if token_first is not None and token_second is not None:
            token = token_first + token_second
            
        return token
    
    def _save_tokens(self, id_token:str, refresh_token:str):
        """ Method to save id and refresh tokens on system's keyring service.
        Windows keyring backend has a length limitation on passwords.
        To avoid this, passwords get splitted up into two credentials.
        """
        
        # save two id_token halves
        id_token_first = id_token[:len(id_token)//2]
        id_token_second = id_token[len(id_token)//2:]
        keyring.set_password(self.keyring_service,'id_token_first', id_token_first)
        keyring.set_password(self.keyring_service,'id_token_second', id_token_second)

        # save refresh_token halves
        refresh_token_first = refresh_token[:len(refresh_token)//2]
        refresh_token_second = refresh_token[len(refresh_token)//2:]
        keyring.set_password(self.keyring_service,'refresh_token_first', refresh_token_first)
        keyring.set_password(self.keyring_service,'refresh_token_second', refresh_token_second)

    def login(self) -> str:
        """ This methods checks if we have a valid (non-expired) id-token
        and returns it, otherwise it gets a new one with refresh-token.
        If refresh-token doesn't exist, it asks user for credentials.
        """
        # check if id_token exists
        id_token = self._get_token('id_token')
        if id_token is None:
            # authenticate against '/login' endpoint
            self._authenticate()
            
            # get id_token
            id_token = self._get_token('id_token')
            
        # check id_token is not expired yet
        expiration_date = jwt.decode(id_token, options={"verify_signature": False}, algorithms=["RS256"])['exp']
        if expiration_date < (datetime.datetime.now(datetime.timezone.utc).timestamp() + self.time_safety_factor):
            print("Your id token is expired. Refreshing...")
            
            # get refresh_token
            refresh_token = self._get_token('refresh_token')
            if refresh_token is not None:
                self._authenticate('refresh')
            else:
                self._authenticate()
                
            # get id_token
            id_token = self._get_token('id_token')
                    
        return id_token

    def recent_jobs(self,
                    start:str = None,
                    end:str = None,
                    days:int = None,
                    jobs:int = None):
        id_token = self.login()
        if start is not None and end is not None:
            res = requests.get(f'{self.url}metering?start={start}&end={end}',
                               headers={'Authorization': id_token})
            self._response_check(res, f'metering between {start} and {end}')
            return res.json()
        elif days is not None:
            res = requests.get(f'{self.url}metering?days={days}',
                               headers={'Authorization': id_token})
            self._response_check(res, f'metering of last {days} days')
            return res.json()
        elif jobs is not None:
            res = requests.get(f'{self.url}metering?jobs={jobs}',
                               headers={'Authorization': id_token})
            self._response_check(res, f'metering of last {jobs} jobs')
            return res.json()
        else:
            raise ValueError('Need more information to make a metering request')

    def submit_job(self,
                   qasm_str: str,
                   shots: int = None,
                   machine: str = None,
                   batch_max: int = None,
                   batch_id: str = None,
                   batch_end: bool = False,
                   options: Optional[dict] = None,
                   name: str = 'job') -> str:
        """
        Submits job to device and returns job ID.
        
        Args:
            qasm_str:   OpenQASM file to run
            shots:      number of repetitions of qasm_str
            machine:    machine to run on
            batch_max:  maximum hqc for a batch request (batching only)
            batch_id:   batch id the job should be part of (batching only)
            batch_end:  explicit batch termination (batching only)
            options:    A dictionary of API options.
            name:       name of job (for error handling)
        
        Returns:
            (str):     id of job submitted
        
        """
        try:
            if not machine and not self.machine:
                raise RuntimeError('Must provide valid machine name')
            # send job request
            body = {
                'machine': machine if machine else self.machine,
                'name': name,
                'language': 'OPENQASM 2.0',
                'program': qasm_str,
                'priority': 'normal',
                'count': shots if shots else self.shots,
                'options': options,
            }
            if batch_max:
                body['batch-exec'] = batch_max

            if batch_id:
                body['batch-exec'] = batch_id

            if batch_end:
                body['batch-end'] = batch_end
                
            id_token = self.login()
            res = requests.post(
                f'{self.url}job',
                json.dumps(body),
                headers={'Authorization': id_token}
            )
            self._response_check(res, 'job submission')

            # extract job ID from response
            jr = res.json()
            job_id = jr['job']
            print(f'submitted {name} id={{job}}, submit date={{submit-date}}'.format(**jr))
            
        except ConnectionError:
            if len(sys.argv) > 2:
                print('{} Connection Error: Error during submit...'.format(name))
            else:
                print('Connection Error: Error during submit...')

        self.recently_submitted.append({'job':job_id,
                                        'name':name,
                                        'machine':machine,
                                        'program':qasm_str})
        return job_id

    def _response_check(self,
                       res,
                       description):
        """Consolidate as much error-checking of response 
        """
        # check if token has expired or is generally unauthorized
        if res.status_code == HTTPStatus.UNAUTHORIZED:
            jr = res.json()
            raise RuntimeError(f'Authorization failure attempting: {description}.\n\nServer Response: {jr}')
        elif res.status_code != HTTPStatus.OK:
            jr = res.json()
            raise RuntimeError(f'HTTP error attempting: {description}.\n\nServer Response: {jr}')

    def retrieve_job_status(self, job_id: str, use_websocket: bool = None, results_format: str = None) -> dict:
        """
        Retrieves job status from device.
        
        Args:
            job_id:        unique id of job
            use_websocket: use websocket to minimize interaction
            results_format: use to format the job results

        Returns:
            (dict):        output from API 
    
        """
        job_url = f'{self.url}job/{job_id}'
        # Using the login wrapper we will automatically try to refresh token
        id_token = self.login()
        if use_websocket or (use_websocket is None and self.use_websocket):
            job_url += '?websocket=true'

        # check if the user is requesting a supported formatting type
        if results_format and results_format.lower() in self.RESULTS_FORMAT:
            #check if we need to add it as first or additional query param
            param_prefix = "?" if "?" not in job_url else "&"
            job_url += param_prefix + 'results_format=' + results_format.lower()

        res = requests.get(job_url, headers={'Authorization': id_token})


        jr = None
        # Check for invalid responses, and raise an exception if so
        self._response_check(res, 'job status')
        # if we successfully got status return the decoded details
        if res.status_code == HTTPStatus.OK:
            jr = res.json()
        return jr
        
        
    def retrieve_job(self, job_id: str, use_websocket: bool = None, results_format: str = None) -> dict:
        """
        Retrieves job from device.
        
        Args:
            job_id:        unique id of job
            use_websocket: use websocket to minimize interaction
            results_format: use to format the job results

        Returns:
            (dict):        output from API
    
        """
        jr = self.retrieve_job_status(job_id, use_websocket, results_format)
        if 'status' in jr and jr['status'] in self.JOB_DONE:
            return jr
        
        if 'websocket' in jr:
            # wait for job completion using websocket
            jr = asyncio.get_event_loop().run_until_complete(self._wait_results(job_id))
        else:
            # poll for job completion
            jr = self._poll_results(job_id)
        return jr

    def _poll_results(self, job_id):
        jr = None
        while True:
            try:
                id_token = self.login()
                jr = self.retrieve_job_status(job_id)

                # If we are failing to retrieve status of any kind, then fail out.
                if jr is None:
                    break;

                if 'status' in jr and jr['status'] in self.JOB_DONE:
                    return jr

                time.sleep(self.retry_timeout)
            except KeyboardInterrupt:
                raise RuntimeError('Keyboard Interrupted')
        return jr

    async def _wait_results(self, job_id):
        while True:
            id_token = self.login()
            jr = self.retrieve_job_status(job_id, True)
            if not jr:
                return jr
            elif 'status' in jr and jr['status'] in self.JOB_DONE: 
                return jr
            else:
                task_token = jr['websocket']['task_token']
                execution_arn = jr['websocket']['executionArn']
                websocket_uri = self.url.replace('https://', 'wss://ws.')
                async with websockets.connect(websocket_uri) as websocket:
                    body = {
                        "action": "OpenConnection",
                        "task_token": task_token,
                        "executionArn": execution_arn
                    }
                    await websocket.send(json.dumps(body))
                    while True:
                        try:
                            res = await asyncio.wait_for(websocket.recv(), timeout=self.ws_timeout)
                            jr = json.loads(res)
                            if 'status' in jr and jr['status'] in self.JOB_DONE:
                                return jr
                        except (asyncio.TimeoutError, websockets.exceptions.ConnectionClosed):
                            try:
                                # Try to keep the connection alive...
                                pong = await websocket.ping()
                                await asyncio.wait_for(pong, timeout=10)
                                continue
                            except:
                                # If we are failing, wait a little while, then start from the top
                                await asyncio.sleep(self.retry_timeout)
                                break
                        except KeyboardInterrupt:
                            raise RuntimeError('Keyboard Interrupted')
            
    def run_job(self,
                qasm_str: str,
                shots: int,
                machine: str,
                name: str = 'job') -> dict:
        """
        Submits a job and waits to receives job result dictionary.
            
        Args:
            qasm_file:  OpenQASM file to run
            name:       name of job (for error handling)
            shots:      number of repetitions of qasm_str
            machine:    machine to run on
            
        Returns:
            jr:         (dict) output from API
        
        """
        job_id = self.submit_job(qasm_str=qasm_str, shots=shots, machine=machine, name=name)
        
        jr = self.retrieve_job(job_id)
        
        return jr


    def status(self, machine: str = None) -> str:
        """
        Check status of machine.
        
        Args:
            (str):    machine name
  
        """
        id_token = self.login()
        res = requests.get(
            f'{self.url}machine/{machine if machine else self.machine}',
            headers={'Authorization': id_token}
        )
        self._response_check(res, 'get machine status')
        jr = res.json()
        
        return jr['state']


    def cancel(self, job_id: str) -> dict:
        """
        Cancels job.
        
        Args:
            job_id:     job ID to cancel

        Returns:
            jr:         (dict) output from API

        """

        id_token = self.login()
        res = requests.post(
            f'{self.url}job/{job_id}/cancel',
            headers={'Authorization': id_token}
        )
        self._response_check(res, 'job cancel')
        jr = res.json()
        
        return jr

    def _mfa_login(self, user_name, pwd):
        """ Allows a user to login with multi-factor authentication enabled"""

        print("Multi-factor authentication (MFA) is enabled.")

        # ask for the mfa verification code
        mfa_code = self._get_mfa_code()

        # prepare the mfa login request
        body = {
            'email': user_name, 
            'password':  pwd, 
            'code': mfa_code
        }
  
        # send login request to API
        status_code, message = self._request_tokens(body)

        # check if they entered a valid mfa code and met all compliance checks
        if status_code != HTTPStatus.OK:

            error = message.get('error')

            # handle the unauthorized code (in case of invalid mfa code or terms and conditions)
            if error['code'] == self.TC_REQUIRED:
                # let them know the mfa code was in fact valid
                print("MFA verification code valid")
                url = message.get('error', {}).get('url', '')
                # let them know what needs to happen in order to have a successful login next time
                print('Terms and conditions must be accepted on UI (' + url + ') before logging in')
                raise RuntimeError(f'Request unauthorized. Terms and conditions required', HTTPStatus.UNAUTHORIZED)

            elif error['code'] == self.MFA_CODE_INVALID:
                print(message.get('error', {}).get('text', 'Invalid or expired verification code. Please try again'))
                raise RuntimeError(f'Invalid or expired verification code.', HTTPStatus.UNAUTHORIZED)

        return status_code, message

    def _federated_login(self):
        """ Allows a user to login by brining their own credentials from an external identity provider """
        
        print("Requesting federated authentication via " + self.provider)

        username, token = None, None

        if self.provider.lower() == self.AZURE_PROVIDER:

            username, token = self._microsoft_login()

        else:
            raise RuntimeError(f'Unsupported provider for login', HTTPStatus.UNAUTHORIZED)

        return username, token


    def _microsoft_login(self):
        """ Allows a user to login via Microsoft Azure Active Directory """

        username, token = None, None
        
        # Create a preferably long-lived app instance which maintains a token cache.
        app = msal.PublicClientApplication(self.AZURE_AD_APP_ID, authority=self.AZURE_AD_AUTHORITY)

        # initiate the device flow authorization. It will expire after 15 minutes
        flow = app.initiate_device_flow(scopes=self.AZURE_AD_SCOPE)
    
        # check if the device code is available in the flow
        if "user_code" not in flow:
            raise ValueError("Fail to create device flow. Err: %s" % json.dumps(flow, indent=4))

        # this prompts the user to visit https://microsoft.com/devicelogin and enter the provided code on a separate browser/device
        code = flow['user_code']
        link = flow['verification_uri']

        print("To sign in:")
        print("1) Open a web browser (using any device)")
        print("2) Visit " + link)
        print("3) Enter code '" + code + "'")
        print("4) Enter your Microsoft credentials")

        # This will block until the we've reached the flow's expiration time 
        result = app.acquire_token_by_device_flow(flow)

        # check if we have an ID Token 
        if "id_token" in result:
            token = result['id_token']
            username = result['id_token_claims']['preferred_username']
           
        else:

            # Check if a timeout occurred
            if 'authorization_pending' in result.get("error"):
                print("Authorization code expired. Please try again.")
            else:
                # some other error occurred
                print(result.get("error"))
                print(result.get("error_description"))
                print(result.get("correlation_id"))  # You may need this when reporting a bug
            
            # a token was not returned (an error occurred or the request timed out)
            raise RuntimeError(f'Unable to authorize federated login', HTTPStatus.UNAUTHORIZED)

        return username, token
